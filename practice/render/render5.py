# Теперь попробуем правильно отрисовать освещение рассеянным и направленным светом.

import numpy as np
from vispy import gloo
from vispy import app

class Surface(object):
    def __init__(self, size=(100,100), nwave=5):
        self._size=size
        self._wave_vector=5*np.random.randn(nwave,2)
        self._angular_frequency=np.random.randn(nwave)
        self._phase=2*np.pi*np.random.rand(nwave)
        self._amplitude=np.random.rand(nwave)/nwave
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
    # считаем нормаль к поверхности как 
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

vert = ("""
#version 120

attribute vec2 a_position;
attribute float a_height;
"""
# Для оценки освещенности нам потребуется нормаль к поверхностию
# Нормаль естественно определить для каждого узла сетки,
# вершины треугольников, отвечающие одному узлу сетки, должны
# иметь одинаковые нормали.
# Передаем только xy координаты, координата z всегда равна 1.
# Система координат для a_position и для a_normal одна.
"""
attribute vec2 a_normal;
"""
# Теперь нам нужно передать направление направленного света.
# Так как это свойство всей сцены, а не отдельной вершины,
# то передаем его в uniform.
# Вектор нормирован.
"""
uniform vec3 u_sun_direction;
"""
# Из шейдера возвращаем освещенность направленным светом для вершины.
"""
varying float v_directed_light;

void main (void) {
    """
    # Считаем косинус угла между нормалью к поверхностью
    # и направлением на источник света
    """
    vec3 normal=normalize(vec3(a_normal, 1));
    v_directed_light=max(0,dot(normal, u_sun_direction));

    float z=(1-a_height)*0.5;
"""
    # Так как теперь мы используем буфер глубины,
    # то нужно установить правильное значение координаты z.
    # Не забываем, что все координаты будут поделены на последнюю.
"""
    gl_Position = vec4(a_position.xy,a_height*z,z);
}
""")

frag = ("""
#version 120
"""
# Передаем цвет рассеянного и направленного источников света
"""
uniform vec3 u_sun_color;
uniform vec3 u_ambient_color;

varying float v_directed_light;

void main() {
    """
    # Цвет получаем суммирование рассянного и направленого света,
    # если яркость слишком большая, просто отсекаем лишнее.
    """
    vec3 rgb=clamp(u_sun_color*v_directed_light+u_ambient_color,0.0,1.0);
    gl_FragColor = vec4(rgb,1);
}
""")

class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface simulator 2")
        # Разрешаем буфер глубины, так как треугольники иногда перекрывются.
        gloo.set_state(clear_color=(0,0,0,1), depth_test=True, blend=False)
        self.program = gloo.Program(vert, frag)
        self.surface=Surface()
        self.program["a_position"]=self.surface.position()
        # Устанавливаем направление на Солнце
        sun=np.array([0,0,1],dtype=np.float32) 
        sun/=np.linalg.norm(sun)
        self.program["u_sun_direction"]=sun
        # Устанавливаем цвета рассеянного света и Солнца
        self.program["u_sun_color"]=np.array([0.8,0.8,0],dtype=np.float32) 
        self.program["u_ambient_color"]=np.array([0.2,0.2,0.5],dtype=np.float32) 
        self.triangles=gloo.IndexBuffer(self.surface.triangulation())
        self.t=0
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.activate_zoom()
        self.show()

    def activate_zoom(self):
        self.width, self.height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)

    def on_draw(self, event):
        gloo.clear()
        self.program["a_height"]=self.surface.height(self.t)
        # Для каждой вершины передаем нормали
        self.program["a_normal"]=self.surface.normal(self.t)
        self.program.draw('triangles', self.triangles)
        # Получаем изображение похожее на атласную ткань.

    def on_timer(self, event):
        self.t+=0.01
        self.update()        

    def on_resize(self, event):
        self.activate_zoom()

if __name__ == '__main__':
    c = Canvas()
    app.run()
