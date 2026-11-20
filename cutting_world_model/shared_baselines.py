"""Current-task frozen MPPI adapters: learned dynamics or nonlearned FDCC surrogate.

The analytic surrogate uses exact robot geometry/action mapping and coarse ideal
joint tracking under the existing FDCC controller, holding the last supplied
wrench-history sample over the horizon. It is NOT a calibrated material/fracture model.
"""
from dataclasses import asdict
import importlib.metadata
import json
import time

import torch

from .control_state import ControlStateSpec
from .checkpoint import checkpoint_identity
from .mpc import ModelRollout
from .free_cut import POSE_DEPTH_VERSIONS
from .pose_depth import pose_depth_spec_class, depth_progress
from .pose_cut import project_board
from .world_model import CuttingWorldModel
from .policy import SliceItCritic

MPPI_REVISION = 'e04a569cd4215e705f9013145c496fc59cb25ed6'


def frame_metadata(env):
    return ControlStateSpec.metadata_from_frame(
        env.metadata, {k: v[0].detach().cpu().numpy() for k, v in env.current.items()})


def validate_current_interface(saved, actual, allow_backend_transfer=False):
    if actual.get('free_cut_task', {}).get('version') not in POSE_DEPTH_VERSIONS:
        raise ValueError('Current baselines require pose_depth_v2/v3')
    # An unchanged task observation interface is not enough when the FDCC
    # sensor frames, gains or compiled robot have changed. Paths may relocate.
    old_platform, new_platform = saved.get('robot_platform'), actual.get('robot_platform')
    if (old_platform is None) != (new_platform is None):
        raise ValueError('Baseline checkpoint/environment mismatch: robot_platform')
    if new_platform is not None:
        for name in ('revision', 'controller_sha256', 'runtime_sha256'):
            if not old_platform.get(name) or not new_platform.get(name):
                raise ValueError('Robot platform checkpoint lacks verified identity: ' + name)
            if old_platform[name] != new_platform[name]:
                raise ValueError('Baseline checkpoint/environment mismatch: robot_platform.' + name)
    keys = ('environment', 'controller', 'policy_observation_revision', 'physical_backend',
            'task_interface_revision', 'tool_revision', 'progress_semantics',
            'board_pose_geometry', 'contact_discretization_revision', 'free_cut_task',
            'state_shapes', 'prediction_layout', 'prediction_history', 'action_bounds',
            'robot_xml', 'robot_base', 'robot_tip', 'joint_names', 'goal_dim')
    transfer = allow_backend_transfer and saved.get("physical_backend") != actual.get("physical_backend")
    allowed = {"physical_backend", "contact_discretization_revision", "board_pose_geometry"} if transfer else set()
    for key in keys:
        if key not in allowed and saved.get(key) != actual.get(key):
            raise ValueError('Baseline checkpoint/environment mismatch: ' + key)
    if transfer:
        if {saved["physical_backend"], actual["physical_backend"]} != {"mujoco_warp_disect", "mujoco_warp_blademaster"}:
            raise ValueError("Unsupported visual cross-backend transfer")
        # Same physical blade and board plane, despite vertex ordering and
        # backend-specific contact solver descriptions. No weight conversion.
        a,b=saved["board_pose_geometry"],actual["board_pose_geometry"]
        va,vb=(torch.as_tensor(x["vertices_in_ee"],dtype=torch.float64) for x in (a,b))
        if va.shape != vb.shape or max(torch.cdist(va,vb).min(0).values.max(),
                                      torch.cdist(va,vb).min(1).values.max()) > 1e-8:
            raise ValueError("Transfer blade geometry mismatch")
        na,nb=(torch.as_tensor(x["board_normal_base"],dtype=torch.float64) for x in (a,b))
        oa,ob=(torch.as_tensor(x["board_origin_base"],dtype=torch.float64) for x in (a,b))
        if not torch.allclose(na,nb,atol=1e-10,rtol=0) or abs(torch.dot(na,oa-ob)) > 1e-8 or a["stop_gap_m"]!=b["stop_gap_m"]:
            raise ValueError("Transfer board plane mismatch")


class AnalyticFDCCModel:
    """No fitted parameters, material labels, particles or fracture progress.

    Controller internal state is reinitialized from q/qd at each coarse model
    step; actual robot inertia/servo and changing contact wrench are approximated.
    Replanning corrects the state with new proprioception/wrench every real step.
    """
    def __init__(self, metadata, controller_parameters, substeps=5, device='cpu'):
        from cutting_gpu_backend.controller.fdcc import SerialChain, ControllerConfig
        if type(substeps) is not int or substeps < 1:
            raise ValueError('Positive analytic substep count required')
        if metadata['free_cut_task']['version'] not in POSE_DEPTH_VERSIONS:
            raise ValueError('Analytic model supports only geometric pose-depth task')
        if 'object_slip' in metadata['free_cut_task']:
            raise ValueError('Analytic model cannot predict particle slip')
        self.spec = pose_depth_spec_class(metadata['free_cut_task']['version'])(metadata).to(device)
        self.chain = SerialChain(metadata['robot_xml'], metadata['robot_base'],
            metadata['robot_tip'], metadata['joint_names']).to(device=device, dtype=torch.float64)
        self.controller_config = ControllerConfig.from_parameters(controller_parameters)
        self.substeps = substeps

    @torch.no_grad()
    def predict(self, state, action, goal):
        from cutting_gpu_backend.controller.fdcc import FDCCController, SlicingActionMapper
        spec = self.spec
        action = spec.execute(action)
        q0 = spec.field(state, 'joint_q').double()
        q = q0.clone()
        controller = FDCCController(self.chain, self.controller_config)
        controller.reset(q, spec.field(state, 'joint_qd').double())
        mapper = SlicingActionMapper(
            max_twist=spec.metadata['controller']['max_twist'],
            robot_control_dt=spec.metadata['environment']['robot_control_dt'],
            pitch_control=spec.metadata['free_cut_task'].get('pitch_control'),
            agent_control_dt=spec.dt)
        mapper.reset(len(q), q.device, q.dtype)
        mapper.added_motion_command = spec.field(state, 'motion_command').double().clone()
        current_pose = self.chain(q, mass_matrix=False)[0][self.chain.tip]
        arguments = mapper(action.double(), goal[..., :7].double(), current_pose)
        # Last supplied chronological history sample; the inherited observation
        # history can lag the full sensor queue. Do not claim zero-delay sensing.
        wrench = spec.field(state, 'wrench_history')[:, -1].double()
        for _ in range(self.substeps):
            command = controller.update(q, sensor_wrench=wrench, **arguments)
            q = controller.enforce_position_limits(command, spec.dt / self.substeps)
        fields = {name: spec.field(state, name).clone()
                  for name in spec.physical_fields + spec.history_fields}
        fields['joint_q'] = q.to(state)
        fields['joint_qd'] = ((q-q0)/spec.dt).to(state)
        fields['peak_wrench'] = wrench.abs().to(state)
        fields['wrench_history'] = wrench.to(state)[:, None].expand_as(fields['wrench_history']).clone()
        v3 = 'last_wrench' in fields
        if v3:
            # v3: the held wrench is also the predicted last force-limit reading.
            fields['last_wrench'] = wrench.to(state)
            fields, limited, end_gap = project_board(spec, state, fields, with_contact=True)
        else:
            fields = project_board(spec, state, fields)
        fields['cut_progress'] = depth_progress(spec.fk(fields['joint_q']),
            spec.board_geometry, fields['depth_span'], spec.field(state, 'cut_progress'))
        if v3:
            fields = spec.apply_board_latch(state, fields, limited, end_gap)
        predicted, reward, done = spec.derive(state, fields, action, goal)
        # Analytic task terminal, no learned survival/head or critic.
        logits = torch.where(done, state.new_full(done.shape, float('inf')),
                             state.new_full(done.shape, -float('inf')))
        return predicted, reward, logits


class SharedMPPIPolicy:
    """Independent receding-horizon plans per world, shared frozen dynamics weights."""
    def __init__(self, env, config, checkpoint=None, controller_parameters=None,
                 analytic_substeps=5, allow_backend_transfer=False):
        from pytorch_mppi import MPPI
        distribution = importlib.metadata.distribution('pytorch-mppi')
        revision = json.loads(distribution.read_text('direct_url.json') or '{}').get('vcs_info', {}).get('commit_id')
        if revision != MPPI_REVISION:
            raise ValueError('Install the pinned planner from requirements/baselines.txt')
        metadata = frame_metadata(env)
        if metadata.get('free_cut_task', {}).get('version') not in POSE_DEPTH_VERSIONS:
            raise ValueError('Shared planner requires pose_depth_v2/v3')
        if config.discount != metadata['free_cut_task']['gamma']:
            raise ValueError('Planner discount must equal current task gamma')
        self.critic = None
        identity = None
        self.weights = None
        self.critic_weights = None
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            validate_current_interface(saved['metadata'], metadata, allow_backend_transfer)
            self.model = CuttingWorldModel(saved['metadata']).to(env.device).eval()
            self.weights = saved['learner']['world_model']
            self.model.load_state_dict(self.weights)
            self.model.requires_grad_(False)
            identity = checkpoint_identity(saved)
            if config.terminal_value == 'critic':
                self.critic = SliceItCritic(self.model.spec.state_dim, self.model.spec.goal_dim).to(env.device).eval()
                self.critic_weights = saved['learner']['critic']
                self.critic.load_state_dict(self.critic_weights)
                self.critic.requires_grad_(False)
        else:
            if config.terminal_value != 'none':
                raise ValueError('Nonlearned MPC cannot use a learned terminal critic')
            if controller_parameters is None:
                raise ValueError('Analytic MPC requires controller parameters')
            self.model = AnalyticFDCCModel(metadata, controller_parameters, analytic_substeps, env.device)
        self.spec = self.model.spec
        self.config = config
        self.worlds = env.worlds
        self.rollouts, self.planners = [], []
        for _ in range(env.worlds):
            rollout = ModelRollout(self.model, config.discount, self.critic)
            planner = MPPI(rollout.dynamics, rollout.running_cost, self.spec.state_dim+3,
                noise_sigma=torch.eye(6, device=env.device)*config.noise_std**2,
                num_samples=config.samples, horizon=config.horizon, device=env.device,
                lambda_=config.temperature, u_min=-torch.ones(6,device=env.device),
                u_max=torch.ones(6,device=env.device),
                U_init=torch.zeros(config.horizon,6,device=env.device),
                terminal_state_cost=rollout.final_cost if self.critic is not None else None,
                step_dependent_dynamics=True)
            self.rollouts.append(rollout)
            self.planners.append(planner)
        self.steps = [0]*env.worlds
        self.plans = []
        self.state = self.spec.pack_batch(env.current).float()
        self.manifest = dict(cross_backend_transfer=bool(checkpoint and saved['metadata']['physical_backend'] != metadata['physical_backend']),
            training_backend=saved['metadata']['physical_backend'] if checkpoint else None,
            execution_backend=metadata['physical_backend'],
            method='wm_mppi' if checkpoint else 'analytic_mpc',
            planner='UM-ARM-Lab/pytorch_mppi', upstream_revision=revision,
            planner_config=asdict(config), checkpoint_identity=identity,
            information='q, qd, signed six-axis sensor history, known target and controller/action history; geometric depth from kinematics',
            material_state_used=False,
            model=('frozen learned model' if checkpoint else
                   'FDCC coarse ideal-joint tracking; measured wrench held constant; no material prediction'),
            analytic_substeps=analytic_substeps if checkpoint is None else None,
            parallelism='independent plans evaluated sequentially; shared dynamics weights')

    @torch.no_grad()
    def act(self, frame, active=None):
        goal = frame['target_pose'].float()
        if self.spec.goal_dim == 8:
            goal = torch.cat((goal,frame['target_speed'].float().reshape(self.worlds,1)),-1)
        actions = torch.zeros((self.worlds,6),device=self.state.device)
        for w,(planner,rollout) in enumerate(zip(self.planners,self.rollouts)):
            if active is not None and not bool(active[w]):
                continue
            if self.state.is_cuda: torch.cuda.synchronize(self.state.device)
            started = time.monotonic()
            initial = rollout.initial(self.state[w:w+1],goal[w:w+1])
            for refinement in range(self.config.refinements):
                u = planner.command(initial,shift_nominal_trajectory=(self.steps[w]>0 and refinement==0))
            if not bool(torch.isfinite(u).all()):
                raise FloatingPointError('Nonfinite planned action')
            actions[w] = self.spec.execute(rollout.task_action(u))
            if self.state.is_cuda: torch.cuda.synchronize(self.state.device)
            self.plans.append(dict(world=w,step=self.steps[w],
                planning_seconds=time.monotonic()-started,
                action=actions[w].cpu().tolist(),
                effective_candidates=float(1/planner.omega.square().sum())))
            self.steps[w] += 1
        return actions

    def observe(self, frame):
        self.state = self.spec.pack_batch_transition(frame,self.state)

    def reset(self, frame, mask):
        self.state = self.spec.reset_batch_history(frame,self.state,mask)
        for w in mask.nonzero().flatten().tolist():
            self.planners[w].U.zero_()
            self.steps[w] = 0

    def unchanged(self):
        model_ok = self.weights is None or all(torch.equal(v.cpu(), self.weights[k])
                                              for k, v in self.model.state_dict().items())
        critic_ok = self.critic_weights is None or all(
            torch.equal(v.cpu(), self.critic_weights[k]) for k, v in self.critic.state_dict().items())
        return model_ok and critic_ok
