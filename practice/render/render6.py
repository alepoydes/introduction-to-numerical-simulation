# Добавим текстуру неба

import numpy as np
from vispy import gloo, app, io

from surface import Surface

VS = ("""
#version 120

attribute vec2 a_position;
attribute float a_height;
attribute vec2 a_normal;

varying vec3 v_normal;
varying vec3 v_position;

void main (void) {
    v_normal=normalize(vec3(a_normal, -1));
    v_position=vec3(a_position.xy,a_height);

    float z=(1-a_height)*0.5;
    gl_Position=vec4(a_position.xy/2,a_height*z,z);
}
""")

FS_triangle = ("""
#version 120
uniform sampler2D u_sky_texture;

varying vec3 v_normal;
varying vec3 v_position;

void main() {
    vec3 eye=vec3(0,0,1);
    vec3 to_eye=normalize(v_position-eye);
    vec3 reflected=normalize(to_eye-2*v_normal*dot(v_normal,to_eye)/dot(v_normal,v_normal));

    vec2 texcoord=0.25*reflected.xy/reflected.z+(0.5,0.5);
    vec3 sky_color=texture2D(u_sky_texture, texcoord).rgb;

    vec3 rgb=sky_color;
    gl_FragColor.rgb = clamp(rgb,0.0,1.0);
    gl_FragColor.a = 1;
}
""")

FS_point = """
#version 120

void main() {
    gl_FragColor = vec4(1,0,0,1);
}
"""

class Canvas(app.Canvas):

    def __init__(self, surface, sky="fluffy_clouds.png"):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface simulator 2")
        self.surface=surface
        self.sky=io.read_png(sky)
        self.program = gloo.Program(VS, FS_triangle)
        self.program_point = gloo.Program(VS, FS_point)
        pos=self.surface.position()
        self.program["a_position"]=pos
        self.program_point["a_position"]=pos
        self.program['u_sky_texture']=gloo.Texture2D(self.sky, wrapping='repeat', interpolation='linear')
        self.triangles=gloo.IndexBuffer(self.surface.triangulation())
        self.t=0
        self.are_points_visible=False
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.activate_zoom()
        self.show()

    def activate_zoom(self):
        self.width, self.height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)

    def on_draw(self, event):
        gloo.set_state(clear_color=(0,0,0,1), blend=False)
        gloo.clear()
        h=self.surface.height(self.t)
        self.program["a_height"]=h
        self.program["a_normal"]=self.surface.normal(self.t)
        gloo.set_state(depth_test=True)
        self.program.draw('triangles', self.triangles)
        if self.are_points_visible:
            self.program_point["a_height"]=h
            gloo.set_state(depth_test=False)
            self.program_point.draw('points')

    def on_timer(self, event):
        self.t+=0.01
        self.update()        

    def on_resize(self, event):
        self.activate_zoom()

    def on_key_press(self, event):
        if event.key=='Escape': self.close()
        elif event.key==' ': self.are_points_visible=not self.are_points_visible

if __name__ == '__main__':
    c = Canvas(Surface())
    app.run()
