# Попробуем получить блики на поверхности.
# Для этого заменим рассеивающую поверхность на отражающую.

import numpy as np
from vispy import gloo
from vispy import app

class Surface(object):
    def __init__(self, size=(100,100), nwave=5, max_height=0.2):
        self._size=size
        self._wave_vector=5*(2*np.random.rand(nwave,2)-1)
        self._angular_frequency=3*np.random.rand(nwave)
        self._phase=2*np.pi*np.random.rand(nwave)
        self._amplitude=max_height*np.random.rand(nwave)/nwave
    def position(self):
        xy=np.empty(self._size+(2,),dtype=np.float32)
        xy[:,:,0]=np.linspace(-1,1,self._size[0])[:,None]
        xy[:,:,1]=np.linspace(-1,1,self._size[1])[None,:]
        return xy
    def height(self, t):
        x=np.linspace(-1,1,self._size[0])[:,None]
        y=np.linspace(-1,1,self._size[1])[None,:]
        z=np.zeros(self._size,dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            z[:,:]+=self._amplitude[n]*np.cos(self._phase[n]+
                x*self._wave_vector[n,0]+y*self._wave_vector[n,1]+
                t*self._angular_frequency[n])
        return z
    def normal(self, t):
        x=np.linspace(-1,1,self._size[0])[:,None]
        y=np.linspace(-1,1,self._size[1])[None,:]
        grad=np.zeros(self._size+(2,),dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            dcos=-self._amplitude[n]*np.sin(self._phase[n]+
                x*self._wave_vector[n,0]+y*self._wave_vector[n,1]+
                t*self._angular_frequency[n])
            grad[:,:,0]+=self._wave_vector[n,0]*dcos
            grad[:,:,1]+=self._wave_vector[n,1]*dcos
        return grad
    def triangulation(self):
        a=np.indices((self._size[0]-1,self._size[1]-1))
        b=a+np.array([1,0])[:,None,None]
        c=a+np.array([1,1])[:,None,None]
        d=a+np.array([0,1])[:,None,None]
        a_r=a.reshape((2,-1))
        b_r=b.reshape((2,-1))
        c_r=c.reshape((2,-1))
        d_r=d.reshape((2,-1))
        a_l=np.ravel_multi_index(a_r, self._size)
        b_l=np.ravel_multi_index(b_r, self._size)
        c_l=np.ravel_multi_index(c_r, self._size)
        d_l=np.ravel_multi_index(d_r, self._size)
        abc=np.concatenate((a_l[...,None],b_l[...,None],c_l[...,None]),axis=-1)
        acd=np.concatenate((a_l[...,None],c_l[...,None],d_l[...,None]),axis=-1)
        return np.concatenate((abc,acd),axis=0).astype(np.uint32)

VS = ("""
#version 120

attribute vec2 a_position;
attribute float a_height;
attribute vec2 a_normal;

varying vec3 v_normal;
varying vec3 v_position;

void main (void) {
"""
    # Нормаль передаем в шейдер фрагментов.
"""
    v_normal=normalize(vec3(a_normal, -1));
"""
    # Вычислим положение точки в пространстве и сохраним его.
"""
    v_position=vec3(a_position.xy,a_height);

    float z=(1-a_height)*0.5;
    gl_Position=vec4(a_position.xy/2,a_height*z,z);
}
""")

FS_triangle = ("""
#version 120
"""
# Мы перенесли вычисление отраженного вектора в шейдер фрагментов,
# так как интерполяция освещенности по вершинам дает слишком грубый результат.
# Нужно учитывать, что это сильно замедляет отрисовку кадра.
"""
uniform vec3 u_sun_direction;
uniform vec3 u_sun_color;
uniform vec3 u_ambient_color;

varying vec3 v_normal;
varying vec3 v_position;

void main() {
"""
    # Вычисляем яркость отраженного света, предполагая, что 
    # камера находится в точке eye.
"""
    vec3 eye=vec3(0,0,1);
    vec3 to_eye=normalize(v_position-eye);
"""
    # Сначала считаем направляющий вектор отраженного от поверхности
    # испущенного из камеры луча.
"""
    vec3 reflected=normalize(to_eye-2*v_normal*dot(v_normal,to_eye)/dot(v_normal,v_normal));
"""
    # Яркость блико от Солнца.
"""
    float directed_light=pow(max(0,-dot(u_sun_direction, reflected)),3);
    vec3 rgb=clamp(u_sun_color*directed_light+u_ambient_color,0.0,1.0);
    gl_FragColor = vec4(rgb,1);
}
""")

FS_point = """
#version 120

void main() {
    gl_FragColor = vec4(1,0,0,1);
}
"""

class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface simulator 2")
        self.program = gloo.Program(VS, FS_triangle)
        self.program_point = gloo.Program(VS, FS_point)
        self.surface=Surface()
        self.program["a_position"]=self.surface.position()
        self.program_point["a_position"]=self.surface.position()
        self.program["u_sun_color"]=np.array([0.8,0.8,0],dtype=np.float32) 
        self.program["u_ambient_color"]=np.array([0.1,0.1,0.5],dtype=np.float32) 
        self.triangles=gloo.IndexBuffer(self.surface.triangulation())
        self.t=0
        self.set_sun_direction()
        self.are_points_visible=False
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.activate_zoom()
        self.show()

    def set_sun_direction(self):
        phi=np.pi*(1+self.t*0.1);
        sun=np.array([np.sin(phi),np.cos(phi),-0.5],dtype=np.float32) 
        sun/=np.linalg.norm(sun)
        self.program["u_sun_direction"]=sun

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
        # Получаем блики от блестящей поверхносьт.
        # Интересно, что в этот раз нам потребовалось заменить
        # только шейдеры, чтобы получить новую содель освещенности.
        if self.are_points_visible:
            self.program_point["a_height"]=h
            gloo.set_state(depth_test=False)
            self.program_point.draw('points')

    def on_timer(self, event):
        self.t+=0.01
        self.set_sun_direction()
        self.update()        

    def on_resize(self, event):
        self.activate_zoom()

    def on_key_press(self, event):
        if event.key=='Escape': self.close()
        elif event.key==' ': self.are_points_visible=not self.are_points_visible

if __name__ == '__main__':
    c = Canvas()
    app.run()
