# Добавим текстуру неба

import numpy as np
from vispy import gloo, app, io

from surface import Surface

VS = ("""
#version 120

uniform float u_eye_height;
uniform mat4 u_world_view;
uniform float u_alpha;
uniform float u_bed_depth;

attribute vec2 a_position;
attribute float a_height;
attribute vec2 a_normal;

varying vec3 v_normal;
varying vec3 v_position;
varying vec3 v_reflected;
varying vec2 v_sky_texcoord;
varying vec2 v_bed_texcoord;
varying float v_reflectance;
varying vec3 v_mask;

void main (void) {
    v_position=vec3(a_position.xy,a_height);
    v_normal=normalize(vec3(a_normal, -1));

    vec4 position_view=u_world_view*vec4(v_position,1);
    float z=1-(1+position_view.z)/(1+u_eye_height);
    gl_Position=vec4(position_view.xy,-position_view.z*z,z);

    vec4 eye_view=vec4(0,0,u_eye_height,1);
    vec4 eye=transpose(u_world_view)*eye_view;
    vec3 from_eye=normalize(v_position-eye.xyz);
    vec3 normal=normalize(-v_normal);
    v_reflected=normalize(from_eye-2*normal*dot(normal,from_eye));
    v_sky_texcoord=0.05*v_reflected.xy/v_reflected.z+vec2(0.5,0.5);

    vec3 cr=cross(normal,from_eye);
    float d=1-u_alpha*u_alpha*dot(cr,cr);
    float c2=sqrt(d);
    vec3 refracted=normalize(u_alpha*cross(cr,normal)-normal*c2);
    float c1=-dot(normal,from_eye);

    float t=(-u_bed_depth-v_position.z)/refracted.z;
    vec3 point_on_bed=v_position+t*refracted;
    v_bed_texcoord=point_on_bed.xy+vec2(0.5,0.5);

    float reflectance_s=pow((u_alpha*c1-c2)/(u_alpha*c1+c2),2);
    float reflectance_p=pow((u_alpha*c2-c1)/(u_alpha*c2+c1),2);
    v_reflectance=(reflectance_s+reflectance_p)/2;

    float diw=length(point_on_bed-v_position);
    vec3 filter=vec3(1,0.5,0.2);
    v_mask=vec3(exp(-diw*filter.x),exp(-diw*filter.y),exp(-diw*filter.z));
}
""")

FS_triangle = ("""
#version 120
uniform sampler2D u_sky_texture;
uniform sampler2D u_bed_texture;
uniform vec3 u_sun_direction;
uniform vec3 u_sun_diffused_color;
uniform vec3 u_sun_reflected_color;

uniform float u_diffused_mult;
uniform float u_bed_mult;
uniform float u_depth_mult;
uniform float u_sky_mult;

varying vec3 v_normal;
varying vec3 v_position;
varying vec3 v_reflected;
varying vec2 v_sky_texcoord;
varying vec2 v_bed_texcoord;
varying float v_reflectance;
varying vec3 v_mask;

void main() {
    vec3 sky_color=texture2D(u_sky_texture, v_sky_texcoord).rgb;
    vec3 bed_color=texture2D(u_bed_texture, v_bed_texcoord).rgb;

    vec3 normal=normalize(v_normal);
    float diffused_intensity=u_diffused_mult*max(0, -dot(normal, u_sun_direction));

    vec3 ambient_water=vec3(0,0.3,0.5);
    vec3 image_color=u_bed_mult*bed_color*v_mask+u_depth_mult*ambient_water*(1-v_mask);

    vec3 rgb=u_sky_mult*sky_color*v_reflectance+image_color*(1-v_reflectance)
        +diffused_intensity*u_sun_diffused_color;
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

def normalize(vec):
    vec=np.asarray(vec, dtype=np.float32)
    return vec/np.sqrt(np.sum(vec*vec,axis=-1))[...,None]

class Canvas(app.Canvas):

    def __init__(self, surface, sky="fluffy_clouds.png", bed="seabed.png"):
        # store parameters
        self.surface=surface
        # read textures
        self.sky=io.read_png(sky)
        self.bed=io.read_png(bed)
        # create GL context
        app.Canvas.__init__(self, size=(600, 600), title="Water surface simulator 2")
        # Compile shaders and set constants
        self.program = gloo.Program(VS, FS_triangle)
        self.program_point = gloo.Program(VS, FS_point)
        pos=self.surface.position()
        self.program["a_position"]=pos
        self.program_point["a_position"]=pos
        self.program['u_sky_texture']=gloo.Texture2D(self.sky, wrapping='repeat', interpolation='linear')
        self.program['u_bed_texture']=gloo.Texture2D(self.bed, wrapping='repeat', interpolation='linear')
        self.program_point["u_eye_height"]=self.program["u_eye_height"]=3;
        self.program["u_alpha"]=0.7;
        self.program["u_bed_depth"]=2;
        self.program["u_sun_direction"]=normalize([0,1,1]);
        self.program["u_sun_diffused_color"]=[1,1,0.5];
        self.program["u_sun_reflected_color"]=[1,0.5,0];
        self.triangles=gloo.IndexBuffer(self.surface.triangulation())
        self.t=0
        # Set up GUI
        self.camera=np.array([0,0,1])
        self.up=np.array([0,1,0])
        self.set_camera()
        self.are_points_visible=False
        self.drag_start=None
        self.diffused_flag=True;
        self.bed_flag=True;
        self.depth_flag=True;
        self.sky_flag=True;
        self.apply_flags();
        # Run everything
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.activate_zoom()
        self.show()

    def apply_flags(self):
        self.program["u_diffused_mult"]=0.5 if self.diffused_flag else 0;
        self.program["u_bed_mult"]=1 if self.bed_flag else 0;
        self.program["u_depth_mult"]=1 if self.depth_flag else 0;
        self.program["u_sky_mult"]=1 if self.sky_flag else 0;

    def set_camera(self):
        rotation=np.zeros((4,4), dtype=np.float32)
        rotation[3,3]=1
        rotation[0,:3]=np.cross(self.up,self.camera)
        rotation[1,:3]=self.up
        rotation[2,:3]=self.camera
        world_view=rotation
        self.program['u_world_view']=world_view.T
        self.program_point['u_world_view']=world_view.T

    def rotate_camera(self, shift):
        right=np.cross(self.up,self.camera)
        new_camera=self.camera-right*shift[0]+self.up*shift[1]
        new_up=self.up-self.camera*shift[0]
        self.camera=normalize(new_camera)
        self.up=normalize(new_up)
        self.up=np.cross(self.camera,np.cross(self.up,self.camera))

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
        elif event.key==' ': 
            self.are_points_visible=not self.are_points_visible
            print("Show lattice vertices:",self.are_points_visible)
        elif event.key=='1': 
            self.diffused_flag=not self.diffused_flag; 
            print("Show sun diffused light:",self.diffused_flag)
            self.apply_flags();
        elif event.key=='2': 
            self.bed_flag=not self.bed_flag; 
            print("Show refracted image of seabed:",self.bed_flag)
            self.apply_flags();
        elif event.key=='3': 
            self.depth_flag=not self.depth_flag; 
            print("Show ambient light in water:",self.depth_flag)
            self.apply_flags();
        elif event.key=='4': 
            self.sky_flag=not self.sky_flag; 
            print("Show reflected image of sky:",self.sky_flag)
            self.apply_flags();

    def screen_to_gl_coordinates(self, pos):
        return 2*np.array(pos)/np.array(self.size)-1

    def on_mouse_press(self, event):
        self.drag_start=self.screen_to_gl_coordinates(event.pos)

    def on_mouse_move(self, event):
        if not self.drag_start is None:
            pos=self.screen_to_gl_coordinates(event.pos)
            self.rotate_camera(pos-self.drag_start)
            self.drag_start=pos
            self.set_camera()
            self.update()

    def on_mouse_release(self,event):
        self.drag_start=None

if __name__ == '__main__':
    c = Canvas(Surface(size=(100,100), nwave=5, max_height=0.2))
    c.measure_fps()
    app.run()
