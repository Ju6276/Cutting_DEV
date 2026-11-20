import numpy as np
import pytest
from cutting_world_model.tool_inertia import validate, combine, transform


def box(mass, sides, com):
    a,b,c=sides
    return validate(mass,com,np.diag([mass*(b*b+c*c)/12,mass*(a*a+c*c)/12,mass*(a*a+b*b)/12]))


def test_combining_parts_preserves_kinetic_energy():
    a=box(2,[.1,.2,.3],[.01,-.05,.1])
    b=box(3,[.2,.3,.4],[-.1,.07,.2])
    whole=combine([a,b])
    v=np.array([.3,-.4,.2]);w=np.array([.5,.7,-.6])
    energy=sum(p["mass_kg"]*np.linalg.norm(v+np.cross(w,p["com_m"]))**2+
               w@np.array(p["inertia_kg_m2"])@w for p in [a,b])
    expected=whole["mass_kg"]*np.linalg.norm(v+np.cross(w,whole["com_m"]))**2
    # Continue the sum separately to make origin vs COM explicit.
    expected += w@np.array(whole["inertia_kg_m2"])@w
    np.testing.assert_allclose(energy,expected,atol=1e-14)


def test_sensor_rotation_preserves_inertial_energy():
    part=box(.17,[.01,.25,.04],[0,.02,.03])
    r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    result=transform(part,r,[.1,0,0])
    np.testing.assert_allclose(result["com_m"],[.08,0,.03])
    w=np.array([.3,.2,-.7])
    np.testing.assert_allclose(w@np.array(part["inertia_kg_m2"])@w,
                               (r@w)@np.array(result["inertia_kg_m2"])@(r@w))


@pytest.mark.parametrize("mass,tensor",[(0,np.eye(3)),(float("nan"),np.eye(3)),(1,np.diag([1,1,3])),(1,-np.eye(3))])
def test_invalid_inertias_are_rejected(mass,tensor):
    with pytest.raises(ValueError):
        validate(mass,[0,0,0],tensor)
