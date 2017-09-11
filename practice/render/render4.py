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
    # Возвращает массив индесов вершин треугольников.
    def triangulation(self):
        # Решетка состоит из прямоугольников с вершинами 
        # A (левая нижняя), B(правая нижняя), С(правая верхняя), D(левая верхняя).
        # Посчитаем индексы всех точек A,B,C,D для каждого из прямоугольников.
        a=np.indices((self._size[0]-1,self._size[1]-1))
        b=a+np.array([1,0])[:,None,None]
        c=a+np.array([1,1])[:,None,None]
        d=a+np.array([0,1])[:,None,None]
        # Преобразуем массив индексов в список (одномерный массив)
        a_r=a.reshape((2,-1))
        b_r=b.reshape((2,-1))
        c_r=c.reshape((2,-1))
        d_r=d.reshape((2,-1))
        # Заменяем многомерные индексы линейными индексами
        a_l=np.ravel_multi_index(a_r, self._size)
        b_l=np.ravel_multi_index(b_r, self._size)
        c_l=np.ravel_multi_index(c_r, self._size)
        d_l=np.ravel_multi_index(d_r, self._size)
        # Собираем массив индексов вершин треугольников ABC, ACD
        abc=np.concatenate((a_l[...,None],b_l[...,None],c_l[...,None]),axis=-1)
        acd=np.concatenate((a_l[...,None],c_l[...,None],d_l[...,None]),axis=-1)
        # Обьединяем треугольники ABC и ACD для всех прямоугольников        
        return np.concatenate((abc,acd),axis=0).astype(np.uint32)

vert = ("""
#version 120

attribute vec2 a_position;
attribute float a_height;
"""
# Чтобы передать данные между шейдерами используем varying
"""
varying float v_z;

void main (void) {
    v_z=(1-a_height)*0.5;
"""
     # Мы сохранили высоту в v_z для правильного окрашивания в
     # в шейдере фрагментов.
     # Так как тепловая карте не трехмерная, то устанавливаем 
     # последнюю координату точки в 1, запрещая перспективу.
     # Координата gl_Position.z используется только для отсечения точек
     # и не влияет на координаты точки на экране.
"""
    gl_Position = vec4(a_position.xy,v_z,1);
}
""")

frag = ("""
#version 120

varying float v_z;

void main() {
"""
    # Так как треугольники одного цвета невозможно отличить,
    # будем окрашивать точку согласно ее высоте.
"""
    vec3 rgb=mix(vec3(1,0.5,0),vec3(0,0.5,1.0),v_z);
    gl_FragColor = vec4(rgb,1);
}
""")

class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, size=(600, 600), title="Water surface simulator 2")
        gloo.set_state(clear_color=(0,0,0,1), depth_test=False, blend=False)
        self.program = gloo.Program(vert, frag)
        self.surface=Surface()
        self.program["a_position"]=self.surface.position()
        # Сохраним вершины треугольников в графическую память.
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
        # Теперь мы отресовываем треугольники, поэтому аргумент "triangles",
        # и теперь нужно передать индексы self.triangles вершин треугольников.
        self.program.draw('triangles', self.triangles)
        # В результате видим тепловую карту высоты столба жидкости,
        # нарисованную на поверхности жидкости.

    def on_timer(self, event):
        self.t+=0.01
        self.update()        

    def on_resize(self, event):
        self.activate_zoom()

if __name__ == '__main__':
    c = Canvas()
    app.run()
