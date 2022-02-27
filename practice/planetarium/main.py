import numpy as np
from pathlib import Path


# Мы используем dataclass, чтобы улучшить читабельность кода. 
# Прочитать про них можно здесь:
# https://realpython.com/python-data-classes/
from dataclasses import dataclass

# Мы будем использовать moderngl библиотеку https://github.com/moderngl/moderngl,
# чтобы упростить работу с OpenGL.
# В отличии от других библиотек, moderngl почти не ограничивает вас в доступе к функциям OpenGL,
# что позволяет создавать очень сложную графику на Python.
# Для установки используйте команду
# pip install moderngl
# или аналогичную.  
import moderngl as mgl

# Библотека moderngl может рисовать графику, но она не знает, куда нужно рисовать.
# Перед вызовом OpenGL нужно создать контекст, чаще всего - это окно, внутри которого мы будем рисовать.
# Так как создание окон делается по разному в разных операционных системах,
# удобнее всего использовать вспомогательную библиотеку, упрощающую этот процесс. 
# Для установки используйте следующую или аналогичную команду:
# pip3 install moderngl-window 
import moderngl_window as mglw

#####################################################################################################################

@dataclass 
class Vector3:
    """
    Трехмерный вектор.
    """
    x: np.float64
    y: np.float64
    z: np.float64

    def asarray(self):
        """
        Преобразует в массив numpy.ndarray, сохраняя декартовы координаты в порядке x, y, z.
        """
        return np.array([self.x, self.y, self.z], dtype=np.float64) 

#####################################################################################################################

@dataclass
class Body:
    """
    Состояние и параметры материальной точки.

    Аргументы:
        pos (Vector3) - координаты тела (м).
        vel (Vector3) - скорость тела (м/c).
        mass (float) - масса тела (кг).
        sprite (int) - номер спрайта для отрисовки тела.
        radius (float) - радиус тела (только для визуализации).
        name (str) - название тела.
    """
    pos: Vector3 
    vel: Vector3
    mass: float
    sprite: int = 0
    radius: float = 1.
    name: str = "unnamed"

#####################################################################################################################

@dataclass
class State:
    """
    Состояние системы из N материальных точек. 

    Аргументы:
        pos - массив координат (м), N x 3.
        vel - массив скоростей (м/с), N x 3.
        mass - массив масс тел (кг), N.
    """
    pos: np.float32
    vel: np.float32
    mass: np.float32

    TOL = 1e-16 # Величина, которую мы добавляем, чтобы избежать деления на ноль.


    @classmethod
    def from_bodies(_cls, bodies):
        def unpack(key, dtype, transform=lambda x:x):
            """Извлекает поле key из всех элементов списка bodies и преобразует в массив типа dtype.""" 
            return np.array([transform(getattr(body,key)) for body in bodies], dtype=dtype)

        pos = unpack(key='pos', dtype=np.float64, transform=lambda x: x.asarray())
        vel = unpack(key='vel', dtype=np.float64, transform=lambda x: x.asarray())
        mass = unpack(key='mass', dtype=np.float64)
        return State(pos=pos, vel=vel, mass=mass)


    def force(self, G):
        """
        Вычисляет массив сил, действующих на каждую материальную точку.
        
        Аргументы:
            G (float) - гравитационная постоянная.
        """
        # Массив с радиус векторами между всеми парами точек.
        delta = self.pos[None] - self.pos[:,None]
        # Кубы расстояний между всеми точками системы.
        dist3 = np.power( np.sum(delta**2, axis=-1), 3/2 )+self.TOL
        # Силы для каждой пары атомов.
        # Произведение констант может быть вычислена один раз при инициализации объекта,
        # но для нас это чрезмерная оптимизация. 
        force_pairwise = G*self.mass[None,:,None]*self.mass[:,None,None]*delta/dist3[:,:,None]
        # Суперпозиция сил, дейстующая на одно тело:
        F = np.sum(force_pairwise, axis=1) 
        return F  

    def rhs(self, G):
        """
        Возвращает правую часть уравнения движения, т.е. скорости изменения каждой из переменных.

        Результат:
            dstate (DState) - скорость изменения величин. Все единицы как в State, разделенные на сек.
        """
        return DState(
            pos=self.vel, # diff(pos,t) = vel.
            vel=self.force(G)/self.mass[:,None], # diff(vel,t) = force/mass.
            mass=np.zeros_like(self.mass) # Массы считаем постоянными.
            )

    # Следующие методы перегружают операции умножения и сложения для объектов State,
    # как для элементов векторного пространства.
    def __add__(self, other):
        if not isinstance(other, DState): return NotImplemented
        # Мы не проверяем количество тел в складываемых системах, так как у нас число тел постоянно.
        # В случае разного числа тел мы получим загадочное исключение, так что для универсального кода
        # эту проверку нужно делать. 
        return State(
            pos=self.pos+other.pos,
            vel=self.vel+other.vel,
            mass=self.mass+other.mass,
            )

    def kinetic_energy(self):
        energy_per_body = 0.5*self.mass*np.sum(self.vel**2, axis=-1)
        return np.sum(energy_per_body)

    def potential_energy(self, G):
        delta = self.pos[None] - self.pos[:,None]
        dist2 = np.sqrt( np.sum(delta**2, axis=-1) )+self.TOL
        energy_per_pair = G*self.mass[None,:]*self.mass[:,None]/dist2
        return np.sum(energy_per_pair)

    def energy(self, G):
        return self.kinetic_energy()+self.potential_energy(G)

#####################################################################################################################

class DState(State):
    """
    Класс полностью повторяющий State, однако мы будем хранить в нем скорости изменения 
    координат и скоростей системы.
    Мы вводим класс DState, чтобы например, нельзя было сложить состояние системы с собой, что не имеет смысла.
    """

    def __add__(self, other):
        if not isinstance(other, DState): return NotImplemented
        return DState(
            pos=self.pos+other.pos,
            vel=self.vel+other.vel,
            mass=self.mass+other.mass,
            )    

    # Умножение приращения на скаляр.
    def __mul__(self, other):
        if not np.isscalar(other):
            return NotImplemented
        return DState(
            pos=self.pos*other,
            vel=self.vel*other,
            mass=self.mass*other,
        )

#####################################################################################################################


class World:
    """
    Класс World содержит описание вселенной в нашей модели, включая все параметры и состояние системы.
    Вселенная содержит в себе только материальные точки, взаимодействующие только гравитационно.
    Каждая точка описывается радиус вектором, вектором скорости и массой.
    Для визуализации используются спрайты, отображаемый размер которых можно явно указать,
    однако этот размер никак не влияет на движение тел.  
    """

    def __init__(self, bodies, G=6.6743015E-11):
        """
            Создает систему из nbodies тел с указанными параметрами.
            Число тел определяется длинной переданных аргументов.
            
            Аргументы:
                bodies ([Body]) - перечень тел, включая их начальное состояние.
                G (float) - гравитационная постоянная (Н м^2/кг^2).
                Остальные аргументы передаются в констуктор предка, который создает окно для отрисовки.

        """
        self._unpack_bodies(bodies)
        self._t = 0. # Начальный момент времени.
        self._G = G 

    @property
    def t(self):
        """Текущее время симуляции. 
        Первоначально время всегда равно 0. 
        На каждом шаге время увеличивается на аргумент dt метода step.
        """
        return self._t

    @property
    def state(self):
        return self._state

    @property 
    def nbodies(self):
        return len(self.radius)

    def _unpack_bodies(self, bodies):
        """
        Преобразует список тел в массивы, каждый из которых хранит только одну характеристику, 
        но для всех тел сразу.

        Аргументы:
                bodies ([Body]) - перечень тел, включая их начальное состояние.
        """
        
        self._state = State.from_bodies(bodies) 

        def unpack(key, dtype):
            """Извлекает поле key из всех элементов списка bodies и преобразует в массив типа dtype.""" 
            return np.array([getattr(body,key) for body in bodies], dtype=dtype)

        # sprite - номер спрайта, N.
        self.sprite = unpack(key='sprite', dtype=np.int32)

        # radius - размер спрайта, N.
        self.radius = unpack(key='radius', dtype=np.float64)

        # names - названия спрайтов
        self.names = list(map(lambda x: x.name, bodies))

    def step(self, dt=1e-3):
        """
        Функция, которую вам нужно переопределить в рамках задания.
        Делает шаг по времени величины dt, обновляя положения и скорости тел согласно уравнениям:

        diff(pos[k], t) = vel[k],
        diff(vel[k], t) = force[k]/mass[k]
        force[k] = sum(, n)

        где через diff(, t) обозначена производная по времени,
            sum(, n) - сумма по n,
            pos[k] и vel[k] суть координаты и скорости тела k и т.п.

        Аргументы:
            dt (float) - шаг по времени (сек). 
        """
        dstate = self._state.rhs(G=self._G)
        self._state = self._state + dstate*dt
        self._t += dt


#####################################################################################################################

class SpriteRender:
    """
    Класс для отрисовки спрайтов.
    """

    VERTEX_SHADER = """
        #version 330
        uniform float u_size;
        uniform vec2 u_center;
        uniform ivec2 u_nsymbols;

        in vec2 in_vert;
        in float in_radius;
        in vec3 in_pos;
        in int in_sprite;

        out vec2 v_tex;

        void main() {
            gl_Position = vec4( (in_vert*in_radius+in_pos.xy-u_center)/u_size, 0.0, 1.0);

            int row = in_sprite / u_nsymbols.x;
            int col = in_sprite - row*u_nsymbols.x;

            v_tex = (vec2(col,row)+(1.0+in_vert)/2.0)/u_nsymbols;
        }
"""

    FRAGMENT_SHADER = """
        #version 330
        uniform sampler2D u_texture;

        in vec2 v_tex;

        out vec4 f_color;
        void main() {
            vec4 texel = texture(u_texture, v_tex);
            f_color = vec4(1-texel.rgb, 1.0);
        }
"""

    def __init__(self, ctx, texture, maxbodies=8):
        # Сохраняем OpenGL контекст, куда мы будем рисовать.
        self.ctx = ctx

        self.prog = self.ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER,
        )

        self.u_center = self.prog['u_center']
        self.u_size = self.prog['u_size']
        self.u_nsymbols = self.prog['u_nsymbols']

        self.u_nsymbols.value = (5,3)

        self.texture = texture

        # Координаты вершин квадрата, олицетворяющего спрайт.
        vertices = np.array([
            -1.0, -1.0,
            1.0, -1.0,
            1.0, 1.0,
            -1.0, 1.0,
        ], dtype='f4')

        # Треугольники, из которых складывается спрайт.
        indices = np.array([0, 1, 2, 0, 2, 3], dtype='i4')

        self.vbo = self.ctx.buffer(vertices) # Буфер вершин.
        self.ibo = self.ctx.buffer(indices) # Буфер индексов вершин.

        self.sprite_bo = self.ctx.buffer(np.zeros(maxbodies,dtype='i4'))
        self.radius_bo = self.ctx.buffer(np.zeros(maxbodies,dtype='f4'))
        self.pos_bo = self.ctx.buffer(np.zeros((maxbodies,3),dtype='f4'))

        # Перечень всех аргументов для программы OpenGL.
        vao_content = [
            (self.vbo, '2f', 'in_vert'), # Вершины в переменную in_vert, 2 координаты с плавающий запятой на значение.
            (self.sprite_bo, 'i/i', 'in_sprite'),
            (self.radius_bo, 'f/i', 'in_radius'),
            (self.pos_bo, '3f/i', 'in_pos'),
        ]

        self.vao = self.ctx.vertex_array(self.prog, vao_content, self.ibo)
        self.nbodies = 0

    def update(self, world):
        """
        Обновляет положения спрайтов согласно состоянию мира.
        """
        self.nbodies = world.nbodies # Реальное число тел в системе.
        self.radius_bo.write(world.radius.astype(np.float32)) # Сохраняем радиусы в буфер.
        pos = world.state.pos # Массив всех цент тел.
        self.pos_bo.write(pos.astype(np.float32)) 
        self.sprite_bo.write(world.sprite.astype(np.int32))

        center = np.mean(pos, axis=0) # Геометрический центр системы.
        self.u_center.value = tuple(center[:2]) 
        size = np.max(np.abs(center[None]-pos)) + np.max(world.radius) # Радиус ящика, в который помещается вся система.
        self.u_size.value = size


    def render(self):
        """
        Рисует спрайты в контекст, переданный в конструктор.
        """
        # Вызываем OpenGL программу, созданную ранее.

        # Для отладки можно считать параметры шейдера следующим образом:
        # print(f"{np.frombuffer(self.pos_bo.read(),dtype=np.float32).reshape((-1,3))=}")
        # print(f"{self.u_center.value=}")

        self.texture.use()     
        self.vao.render(instances=self.nbodies)


#####################################################################################################################

class Application(mglw.WindowConfig):
    """
    Класс Application занимается выполнение симуляции и отрисовкой графики. 
    """

    gl_version = (3, 3) # Минимальная требуемая версия OpenGL.
    window_size = (800, 800) # Размер окна.
    resource_dir = (Path(__file__).parent / 'resources').resolve()


    def __init__(self, **kwargs):
        """
        Создает главное окно приложения и запускает симуляцию.
        """
        super().__init__(**kwargs)

        # Создаем описание Солнечной системы.
        solar_system = [
            Body(name='Sol', pos=Vector3(0,0,0), vel=Vector3(0,0,0), mass=1, sprite=10, radius=0.1),
            Body(name='Mars', pos=Vector3(1,0,0), vel=Vector3(0,1,0), mass=.1, sprite=5, radius=0.1),
            Body(name='Earth', pos=Vector3(0,1,0), vel=Vector3(-1,0,0), mass=.1, sprite=13, radius=0.1),
        ]
        # создаем симуляцию.
        self.world = World(
            bodies = solar_system
        ) 
        # Резервируем переменные под рендеры, но не создаем их, так как контекст еще не проинициализирован.
        self.sprites_render = None

    def update_physics(self):
        """
        Делает один шаг симуляции
        """
        self.world.step()

    def render(self, _time, _frametime):
        """
        Функция вызывается каждый раз, когда нужно отрисовать содержимое окна.
    
        Аргументы: 
            _time - реальное время (сек),
            _frametime - реальное время, прошедшее с отрисовки предыдущего кадра (сек). 
        """
        # Мы будем игнорировать реальное приращение времени,
        # так как у нес нет задачи соотнести время симуляции и реальное время.
        # Мы рисуем графику только для иллюстрации.

        # Здесь мы вызываем метод для симуляции состояния системы.
        # Это приемлемо, только если вычисления занимают очень мало времени.
        # В противном случае нужно создавать поток, в котором будут производиться расчеты,
        # а в главном потоке будет только отрисовываться графика. 
        self.update_physics()

        # Здесь начинается отрисовка.
        # Сначала мы очищаем экран. 
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.viewport = (0,0)+self.window_size

        # Теперь рисуем небесные тела.
        # Создаем рендер, если это еще не сделано.
        if not self.sprites_render:
            print(f"Resource directory: {self.resource_dir}")
            texture = self.load_texture_2d('symbols.jpg')
            self.sprites_render = SpriteRender(ctx=self.ctx, texture=texture)

        self.sprites_render.update(world=self.world)
        self.sprites_render.render()

#####################################################################################################################


if __name__ == '__main__':
    # Создаем окно приложения и запускаем симуляцию.
    mglw.run_window_config(Application)
