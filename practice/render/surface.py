import numpy as np

class PlaneWaves(object):
    def __init__(self, size=(100,100), nwave=5, max_height=0.2):
        self._size=size
        self._wave_vector=5*(2*np.random.rand(nwave,2)-1)
        self._angular_frequency=2*np.random.rand(nwave)
        self._phase=2*np.pi*np.random.rand(nwave)
        self._amplitude=max_height*(1+np.random.rand(nwave))/2/nwave
        self.t=0
    def position(self):
        xy=np.empty(self._size+(2,),dtype=np.float32)
        xy[:,:,0]=np.linspace(-1,1,self._size[0])[:,None]
        xy[:,:,1]=np.linspace(-1,1,self._size[1])[None,:]
        return xy
    def propagate(self, dt):
        self.t+=dt
    def height_and_normal(self):
        x=np.linspace(-1,1,self._size[0])[:,None]
        y=np.linspace(-1,1,self._size[1])[None,:]
        z=np.zeros(self._size,dtype=np.float32)
        grad=np.zeros(self._size+(2,),dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            arg=self._phase[n]+x*self._wave_vector[n,0]+y*self._wave_vector[n,1]+self.t*self._angular_frequency[n]
            z[:,:]+=self._amplitude[n]*np.cos(arg)
            dcos=-self._amplitude[n]*np.sin(arg)
            grad[:,:,0]+=self._wave_vector[n,0]*dcos
            grad[:,:,1]+=self._wave_vector[n,1]*dcos
        return z, grad
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

class Surface(PlaneWaves):
    pass