# Добавим текстуру неба

import numpy as np
from vispy import gloo, app, io

from surface import Surface

VS = ("""
#version 120

uniform float u_eye_height;

attribute vec2 a_position;
attribute float a_height;
attribute vec2 a_normal;

varying vec3 v_normal;
varying vec3 v_position;

void main (void) {
    v_normal=normalize(vec3(a_normal, -1));
    v_position=vec3(a_position.xy,a_height);

    // [-1,u_eye_height] -> [1,0] 
    float z=1-(1+a_height)/(1+u_eye_height);
    gl_Position=vec4(a_position.xy,a_height*z,z);
}
""")

FS_triangle = ("""
#version 120
uniform sampler2D u_sky_texture;
uniform sampler2D u_bed_texture;
uniform float u_alpha;
uniform float u_bed_depth;
uniform float u_eye_height;

varying vec3 v_normal;
varying vec3 v_position;

void main() {
    vec3 eye=vec3(0,0,u_eye_height);
    vec3 from_eye=normalize(v_position-eye);
    vec3 normal=normalize(-v_normal);
    vec3 reflected=normalize(from_eye-2*normal*dot(normal,from_eye));

    vec2 sky_texcoord=0.25*reflected.xy/reflected.z+vec2(0.5,0.5);
    vec3 sky_color=texture2D(u_sky_texture, sky_texcoord).rgb;

    vec3 cr=cross(normal,from_eye);
    float d=1-u_alpha*u_alpha*dot(cr,cr);
    float c2=sqrt(d);
    vec3 refracted=normalize(u_alpha*cross(cr,normal)-normal*c2);
    float c1=-dot(normal,from_eye);

    float t=(-u_bed_depth-v_position.z)/refracted.z;
    vec3 point_on_bed=v_position+t*refracted;
    vec2 bed_texcoord=point_on_bed.xy+vec2(0.5,0.5);
    vec3 bed_color=texture2D(u_bed_texture, bed_texcoord).rgb;

    float reflectance_s=pow((u_alpha*c1-c2)/(u_alpha*c1+c2),2);
    float reflectance_p=pow((u_alpha*c2-c1)/(u_alpha*c2+c1),2);
    float reflectance=(reflectance_s+reflectance_p)/2;

    float diw=length(point_on_bed-v_position);
    vec3 filter=vec3(1,0.5,0.2);
    vec3 mask=vec3(exp(-diw*filter.x),exp(-diw*filter.y),exp(-diw*filter.z));
    vec3 ambient_water=vec3(0,0.6,0.8);
    vec3 image_color=bed_color*mask+ambient_water*(1-mask);

    vec3 rgb=sky_color*reflectance+image_color*(1-reflectance);
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

    def __init__(self, surface, sky="fluffy_clouds.png", bed="seabed.png"):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface simulator 2")
        self.surface=surface
        self.sky=io.read_png(sky)
        self.bed=io.read_png(bed)
        self.program = gloo.Program(VS, FS_triangle)
        self.program_point = gloo.Program(VS, FS_point)
        pos=self.surface.position()
        self.program["a_position"]=pos
        self.program_point["a_position"]=pos
        self.program['u_sky_texture']=gloo.Texture2D(self.sky, wrapping='repeat', interpolation='linear')
        self.program['u_bed_texture']=gloo.Texture2D(self.bed, wrapping='repeat', interpolation='linear')
        self.program_point["u_eye_height"]=self.program["u_eye_height"]=3;
        self.program["u_alpha"]=0.9;
        self.program["u_bed_depth"]=1;
        self.triangles=gloo.IndexBuffer(self.surface.triangulation())
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
        h,grad=self.surface.height_and_normal()
        self.program["a_height"]=h
        self.program["a_normal"]=grad
        gloo.set_state(depth_test=True)
        self.program.draw('triangles', self.triangles)
        if self.are_points_visible:
            self.program_point["a_height"]=h
            gloo.set_state(depth_test=False)
            self.program_point.draw('points')

    def on_timer(self, event):
        self.surface.propagate(0.01)
        self.update()        

    def on_resize(self, event):
        self.activate_zoom()

    def on_key_press(self, event):
        if event.key=='Escape': self.close()
        elif event.key==' ': self.are_points_visible=not self.are_points_visible

if __name__ == '__main__':
    c = Canvas(Surface(nwave=5, max_height=0.3))
    app.run()
