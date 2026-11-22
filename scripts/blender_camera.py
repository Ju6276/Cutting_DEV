"""Blender pinhole camera setup shared by studio rendering and projection tests."""
from mathutils import Matrix

def configure_sensor_camera(scene, camera, sensor):
    transform=Matrix(sensor["base_from_camera"])
    # Optical x-right/y-down/z-forward -> Blender x-right/y-up/z-back.
    for row in range(3):
        transform[row][1]*=-1
        transform[row][2]*=-1
    camera.matrix_world=transform
    camera.data.type="PERSP"
    camera.data.sensor_fit="HORIZONTAL"
    camera.data.sensor_width=36.
    camera.data.lens=sensor["fx"]*36./sensor["width"]
    scene.render.resolution_x=sensor["width"]
    scene.render.resolution_y=sensor["height"]
    scene.render.resolution_percentage=100
    ratio=sensor["fx"]/sensor["fy"]
    scene.render.pixel_aspect_x=max(1.,1./ratio)
    scene.render.pixel_aspect_y=max(1.,ratio)
    # Blender 2.82's camera view projection: positive X shift moves the
    # principal point right, negative Y shift moves it down in image pixels.
    camera.data.shift_x=(sensor["cx"]+.5-sensor["width"]/2)/sensor["width"]
    camera.data.shift_y=-(sensor["cy"]+.5-sensor["height"]/2)*ratio/sensor["width"]
    camera.data.clip_start=.01
    camera.data.clip_end=2.
