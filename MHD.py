import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

plt.style.use('test1')

G = 6.674/10**8
Ms = 1.986*10**33
k = 1.381/10**16
pc = 3.08568*10**18
yr =365.0*24.0*3600.0
mh2 = 3.32/10**24
mp = 1.6726/10**24
sos = 0.2*10**5

def H(lam, t = 0):
    if t==0:
        H = (2-lam+(lam**2-4.0*lam)**0.5)/2.0
    else:
        H = (2-lam-(lam**2-4.0*lam)**0.5)/2.0
    n = (1-H+2*lam)/lam
    l = [H, n]
    return l
        
def instruction():
	print('-'*80)
	print("Class begin(lambda, type, initial point, initial conditions(dictionary))\n")
	print("type = 0: from infinity, d = {'A':, 'V':}\n")
	print("type = 1: from near the origin, d = {'D':}\n")
	print("type = 2: type1 solution of crossing the MSC, d = {'dx':}\n")
	print("type = 3: type2 solution of crossing the MSC, d = {'dx':}\n")
	print("type = 4: from anywhere, d = {'a':, 'v':}\n")
	print("type = 5: void expansion, d = {'a0':}\n")
	print("type = 6: free fall, d = {'m':, 'L':}\n")
	print("type = 7: MHDS, d = {'K':, 't':}\n")
	print("begin.to(ending point, steps, num)\n")
	print("SIS(self, x2, k = 1, dx = -1.0/10**6, dxx = -0.01, x1 = 20.0, n = 20000, num = 1000)")
	print('-'*80)

def locus():
	print('-'*80)
	print('locus1(lam, t, x0, xm, s, l, d, n = 0, m = 10000)\n')
	print('locus2(lam, t, xm, l, d, c = 0, n = 0, mindx = 1.0/10**5, m0 = 10000)\n')
	print('locus3(lam, t, xm, l, d, n = 0, m)\n')
	print('locus_s1(lam, t, x0, xm, l, d = {}, h = 0, n, mindx, m0)\n')
	print('locus_s2(lam, t, x0, xm, l, d, X = 1.0/1.5, n, k = 0, f = 1, mindx, m0)\n')
	print('locus_s11(lam, t, xs, xm, l, d, h, n, mindx, m0)\n')
	print('locus_s21(lam, t, xsu/d, xm, l, d, X, n, k, f, mindx, m0\n')
	print('locus_s22(lam, t, x0, xsu/d, xm, l, d, n, k, f, mindx, m0)\n')
	print('-'*80)

class begin:
	"""The class of self-similar solutions."""
	def __init__(self, l = 0.1, t = 0, x = 10.0, d = {'dx':1.0/10**3}):
		self.lam = l
		self.type = t
		self.x1 = x
		self.number = 0
		self.results = []
		if t==0:
			self.a1 = fia(x, d['A'], d['V'], l)
			self.v1 = fiv(x, d['A'], d['V'], l)
		elif t==1:
			self.a1 = foa(x, d['D'], l)
			self.v1 = fov(x, d['D'], l)
		elif t==2:
			self.x1 = x+d['dx']
			self.a1 = fc(x, d['dx'],1, l, 1)
			self.v1 = fc(x, d['dx'],1, l, 2)
		elif t==3:
			self.x1 = x+d['dx']
			self.a1 = fc(x, d['dx'],2, l, 1)
			self.v1 = fc(x, d['dx'],2, l, 2)
		elif t==4:
			self.a1 = d['a']
			self.v1 = d['v']
		elif t==5:
			self.v1 = x
			self.a1 = d['a0']
		elif t==6:
			self.a1 = ffa(x, d['m'], d['L'])
			self.v1 = ffv(x, d['m'], d['L'])
		elif t==7:
			self.a1 = fma(x, d['K'], l, d['t'])
			self.v1 = fmv(x, d['K'], l, d['t'])
			
	def to(self, x = 0.01, n = 10000, num = 1000, c = 0):
		self.number = self.number + 1
		d1 = {'l':self.lam, 'x1':self.x1, 'x2':float(x), 'a':self.a1, 'v':self.v1, 'n':n}
		d2 = RungeKutta(d1, num)
		self.results.append(d2)
		if c==1:
			self.x1 = d2['x'][d2['n']-1]
			self.a1 = d2['a'][d2['n']-1]
			self.v1 = d2['v'][d2['n']-1]
		elif c==2:
			self.x1 = d2['x'][0]
			self.a1 = d2['a'][0]
			self.v1 = d2['v'][0]
		return d2
	def SIS(self, x2, k = 1, dx = -1.0/10**6, dxx = -0.01, x1 = 20.0, n = 20000, num = 1000):
		self.number = self.number + 1
		if k==0:
			if x2>(1+2*self.lam)**0.5:
				d1 = {'l':self.lam, 'x1':x1, 'x2':x2, 'a':fia(x1, 2.0, 0.0, self.lam), 'v':fiv(x1, 2.0, 0.0, self.lam), 'n':n}
				d = RungeKutta(d1, num)
			else:
				d1 = {'l':self.lam, 'x1':x1, 'x2':(1+2*self.lam)**0.5, 'a':fia(x1, 2.0, 0.0, self.lam), 'v':fiv(x1, 2.0, 0.0, self.lam), 'n':n}
				d1_ = RungeKutta(d1, num)
				d2_ = {}
				d2_ = locus2(self.lam,2,x2,[(1.0+2*self.lam)**0.5+dxx,1],{'dx':dx},1,0)['d'][0]
				if d2_ == {}:
					return {}
				d = connect(d2_, d1_)
		else:
			if x2>(1+2*self.lam)**0.5:
				d1 = {'l':self.lam, 'x1':x1, 'x2':x2, 'a':fia(x1, 2.0, 0.0, self.lam), 'v':fiv(x1, 2.0, 0.0, self.lam), 'n':n}
				d = RungeKutta(d1, num)
			else:
				d1 = {'l':self.lam, 'x1':x1, 'x2':(1.0+2*self.lam)**0.5+dx, 'a':fia(x1, 2.0, 0.0, self.lam), 'v':fiv(x1, 2.0, 0.0, self.lam), 'n':n}
				d1_ = RungeKutta(d1, num)
				d2_ = {}
				d2_ = locus2(self.lam,3,x2,[(1.0+2*self.lam)**0.5+dxx,1],{'dx':dx},1,0)['d'][0]
				if d2_ == {}:
					return {}
				d = connect(d2_, d1_)
		self.results.append(d)
		return d
	
def fmv(x, K, lam, t):
	if t==0:
		out = x*(2-lam+(lam**2-4*lam)**0.5)/2
	elif t==1:
		out = x*(2-lam-(lam**2-4*lam)**0.5)/2
	return out
	
def fma(x, K, lam, t):
	if t==0:
		H = (2-lam+(lam**2-4*lam)**0.5)/2
	elif t==1:
		H = (2-lam-(lam**2-4*lam)**0.5)/2
	n = (1-H+2*lam)/lam
	return K*x**(-n)
			
def ffa(x, m, L):
	out = (m/2/x**3)**0.5 - math.log(x, math.e)*3*(2/m/x)**0.5/8 - L/x**0.5
	return out
			
def ffv(x, m, L):
	out = -(2*m/x)**0.5 - math.log(x, math.e)*3*(2*x/m)**0.5/4 -2*L*x**0.5
	return out
			
def fia(x, A, V, lam):
	out = A/x**2 + A*(2-A)/2/x**4 + A*(4-A)*V/3/x**5 + ((lam*(2-A)*A**2)/4 + A*(6-A)*V**2/4 + A*(A-3)*(A/2-1))/x**6
	return out
	
def fiv(x, A, V, lam):
	out = V + (2-A)/x + V/x**2 + ((A-2)*(A/6-1)+2*V**2/3+lam*A*(2-A)/3)/x**3
	return out
	
def foa(x, D, lam):
	out = D + 0.5*(2*D/9-D**2/3-2*lam*D**2)*x**2
	return out
	
def fov(x, D, lam):
	out = 2*x/3 + (2/27-D/9-2*lam*D/3)*x**3/5
	return out
	
def fcv(x, lam):
	if x <= 1.0/27**0.5/lam:
		out = -2*math.cos(math.acos(x*lam*27**0.5)/3)/3**0.5 + x
	else:	
		out = -(x*lam-((x*lam)**2-1.0/27)**0.5)**(1.0/3)-(x*lam+((x*lam)**2-1.0/27)**0.5)**(1.0/3)+x
	return out
	
def fca(x, lam):
	out = 2/x/(x-fcv(x, lam))
	return out
	
def fc(x, dx, t, lam, l):
	v, a = fcv(x, lam), fca(x, lam)
	A, B, C = 2*((v-x)-x*lam/(v-x)**2), 2*(x-v), -2*v/x**2
	delta = (B**2-4*A*C)**0.5
	r1, r2 = (-B + delta)/2/A, (-B - delta)/2/A
	if abs(r1)<abs(r2):
		dv = [r1, r2]
	else:
		dv = [r2, r1]
	da = [(2*(x-v)/x-m)*a/(v-x) for m in dv]
	if t==1:
		if l==1:
			return a+da[0]*dx
		else:
			return v+dv[0]*dx
	else:
		if l==1:
			return a+da[1]*dx
		else:
			return v+dv[1]*dx
	
def f1(x, a, v, lam):
	out = (x-v)*(a*(x-v)-2/x)/((x-v)**2-(1+lam*a*x*x))
	return out
	
def f2(x, a, v, lam):
	out = a*((x-v)*(a-2*(x-v)/x)+2*lam*x*a)/((x-v)**2-(1+lam*a*x*x))
	return out
	
def shock1(x, l, lam, k = 0):
	"""One temperature shock condition given up(down)stream"""
	b1 = 2.0/lam/l[1]/x**2
	m1 = l[0]-x
	X = 1.0
	roots = np.roots([1.0, b1+1.0, -b1*m1**2])
	for i in range(2):
		if roots[i]>0:
			X = roots[i]
	if k==0:
		if X <=1:
			print("X should be larger than 1.\n")
			return []
	else:
		if X >=1:
			print("X should be smaller than 1.\n")
			return []
	a_ = X*l[1]
	v_ = (l[0]-x)/X+x
	return [x,v_,a_,X]

def shock2(x, l, lam, t):
	"""Two temperature shock condition given downstream"""
	if t >=1:
		print("t should be less than 1.\n")
		return []
	b1 = 2.0/lam/l[1]/x**2
	m1 = l[0]-x
	X = 1.0
	x_ = x/t
	roots = np.roots([1.0, b1*t**2, -(b1*m1**2+b1+1.0), b1*m1**2])
	for i in range(3):
		if (roots[i]<1)and(roots[i]>0):
			X = roots[i]
	if X >=1:
		print("Error.\n")
		return []
	a_ = X*l[1]
	v_ = (l[0]-x)/X/t+x_
	return [x_,v_,a_,X]

def shock2_(x, l, lam, t, k = 1):
	"""Two temperature shock condition given upstream"""
	r = []
	j = 0.0
	if t <=1:
		print("t should be larger than 1.\n")
		return []
	b1 = 2.0/lam/l[1]/x**2
	m1 = l[0]-x
	X = 1.0
	x_ = x/t
	XX = (-b1*t**2+(b1**2*t**4+3*(1+b1+b1*m1**2))**0.5)/3
	f = XX**3+b1*t**2*XX**2-(1.0+b1+b1*m1**2)*XX+b1*m1**2
	if f>0:
		print("There is no roots")
		return []
	roots = np.roots([1.0, b1*t**2, -(b1*m1**2+b1+1.0), b1*m1**2])
	for i in range(3):
		if roots[i]>1:
			r.append(roots[i])
	if r==[]:
		print("There is no roots")
		return []
	if r[0]>r[1]:
		j = r[1]
		r[1] = r[0]
		r[0] = j
		#print("Amazing.\n")
	if k==1:
		X = r[0]
	else:
		X = r[1]
	if X <=1:
		print("Error.\n")
		return []
	a_ = X*l[1]
	v_ = (l[0]-x)/X/t+x_
	return [x_,v_,a_,X]

def RungeKutta(d, num = 1000):
	"""RungeKutta Method for numerical integration with optimization near the origin."""
	t = d['x1']
	a = d['a']
	v = d['v']
	lt = []
	la = []
	lv = []
	lm = []
	lb = []
	lvA = []
	lvc = []
	lac = []
	lma = []
	lmv = []
	lam = d['l']
	v_ = fcv(t, lam)
	a_ = fca(t, lam)
	n = d['n']
	lac.append(a_)
	lvc.append(v_)
	lt.append(t)
	la.append(a)
	lv.append(v)
	lm.append(a*t*t*(t-v))
	lb.append((lam*a*a*t*t)**0.5)
	lvA.append(t*(a*lam)**0.5)
	if d['x1']<0.01:
		if d['x2']>0.01:
			B = math.pow(0.01/d['x1'],1.0/num)
			for i in range(num):
				step = t*B-t
				k1 = f1(t, a, v, lam)
				g1 = f2(t, a, v, lam)
				k2 = f1(t+step/2,a+step*g1/2,v+step*k1/2,lam)
				g2 = f2(t+step/2,a+step*g1/2,v+step*k1/2,lam)
				k3 = f1(t+step/2,a+step*g2/2,v+step*k2/2,lam)
				g3 = f2(t+step/2,a+step*g2/2,v+step*k2/2,lam)
				k4 = f1(t+step,a+step*g3,v+step*k3,lam)
				g4 = f2(t+step,a+step*g3,v+step*k3,lam)
				t = t+step
				v = v+step*(k1+2*k2+2*k3+k4)/6
				a = a+step*(g1+2*g2+2*g3+g4)/6
				v_ = fcv(t, lam)
				a_ = fca(t, lam)
				lac.append(a_)
				lvc.append(v_)
				lt.append(t)
				la.append(a)
				lv.append(v)
				lm.append(a*t*t*(t-v))
				lmv.append(t*v**2/2)
				lma.append(2*t**3*a**2)
				lb.append((lam*a*a*t*t)**0.5)
				lvA.append(t*(a*lam)**0.5)
			n = n+num
		else: 
			B = math.pow(d['x1']/d['x2'],1.0/num)
			for i in range(num):
				step = t/B-t
				k1 = f1(t, a, v, lam)
				g1 = f2(t, a, v, lam)
				k2 = f1(t+step/2,a+step*g1/2,v+step*k1/2,lam)
				g2 = f2(t+step/2,a+step*g1/2,v+step*k1/2,lam)
				k3 = f1(t+step/2,a+step*g2/2,v+step*k2/2,lam)
				g3 = f2(t+step/2,a+step*g2/2,v+step*k2/2,lam)
				k4 = f1(t+step,a+step*g3,v+step*k3,lam)
				g4 = f2(t+step,a+step*g3,v+step*k3,lam)
				t = t+step
				v = v+step*(k1+2*k2+2*k3+k4)/6
				a = a+step*(g1+2*g2+2*g3+g4)/6
				v_ = fcv(t, lam)
				a_ = fca(t, lam)
				lac.append(a_)
				lvc.append(v_)
				lt.append(t)
				la.append(a)
				lv.append(v)
				lm.append(a*t*t*(t-v))
				lmv.append(t*v**2/2)
				lma.append(2*t**3*a**2)
				lb.append((lam*a*a*t*t)**0.5)
				lvA.append(t*(a*lam)**0.5)
			n = num
			v_a = [v, a]
			v_a_c = [v_, a_]
			if step<0:
				lac.reverse()
				lvc.reverse()
				lt.reverse()
				la.reverse()
				lv.reverse()
				lm.reverse()
				lma.reverse()
				lmv.reverse()
				lb.reverse()
				lvA.reverse()
			do = {'n':n+1, 'x':lt, 'a':la, 'v':lv, '-v':[-x for x in lv], 'm':lm, 'mv':lmv, 'ma':lma, 'b':lb, 'vA':lvA, 'v-a':v_a, 'l':lam, 'vc':lvc, '-vc':[-x for x in lvc], 'ac':lac, 'v-a-c':v_a_c}
			return do
	if d['x2']<0.01:
		step = (0.01-d['x1'])/d['n']
	elif d['x1']<0.01:
		step = (d['x2']-0.01)/d['n']
	else:
		step = (d['x2']-d['x1'])/d['n']
	for i in range(d['n']):
		k1 = f1(t, a, v, lam)
		g1 = f2(t, a, v, lam)
		k2 = f1(t+step/2,a+step*g1/2,v+step*k1/2,lam)
		g2 = f2(t+step/2,a+step*g1/2,v+step*k1/2,lam)
		k3 = f1(t+step/2,a+step*g2/2,v+step*k2/2,lam)
		g3 = f2(t+step/2,a+step*g2/2,v+step*k2/2,lam)
		k4 = f1(t+step,a+step*g3,v+step*k3,lam)
		g4 = f2(t+step,a+step*g3,v+step*k3,lam)
		t = t+step
		v = v+step*(k1+2*k2+2*k3+k4)/6
		a = a+step*(g1+2*g2+2*g3+g4)/6
		v_ = fcv(t, lam)
		a_ = fca(t, lam)
		lac.append(a_)
		lvc.append(v_)
		lt.append(t)
		la.append(a)
		lv.append(v)
		lm.append(a*t*t*(t-v))
		lmv.append(t*v**2/2)
		lma.append(2*t**3*a**2)
		lb.append((lam*a*a*t*t)**0.5)
		lvA.append(t*(a*lam)**0.5)
	if d['x2']<0.01:
		A = math.pow(0.01/d['x2'],1.0/num)
		for i in range(num):
			step = t/A-t
			k1 = f1(t, a, v, lam)
			g1 = f2(t, a, v, lam)
			k2 = f1(t+step/2,a+step*g1/2,v+step*k1/2,lam)
			g2 = f2(t+step/2,a+step*g1/2,v+step*k1/2,lam)
			k3 = f1(t+step/2,a+step*g2/2,v+step*k2/2,lam)
			g3 = f2(t+step/2,a+step*g2/2,v+step*k2/2,lam)
			k4 = f1(t+step,a+step*g3,v+step*k3,lam)
			g4 = f2(t+step,a+step*g3,v+step*k3,lam)
			t = t+step
			v = v+step*(k1+2*k2+2*k3+k4)/6
			a = a+step*(g1+2*g2+2*g3+g4)/6
			v_ = fcv(t, lam)
			a_ = fca(t, lam)
			lac.append(a_)
			lvc.append(v_)
			lt.append(t)
			la.append(a)
			lv.append(v)
			lm.append(a*t*t*(t-v))
			lmv.append(t*v**2/2)
			lma.append(2*t**3*a**2)
			lb.append((lam*a*a*t*t)**0.5)
			lvA.append(t*(a*lam)**0.5)
		n = n+num
	v_a = [v, a]
	v_a_c = [v_, a_]
	if step<0:
		lac.reverse()
		lvc.reverse()
		lt.reverse()
		la.reverse()
		lv.reverse()
		lm.reverse()
		lma.reverse()
		lmv.reverse()
		lb.reverse()
		lvA.reverse()
	do = {'n':n+1, 'x':lt, 'a':la, 'v':lv, '-v':[-x for x in lv], 'm':lm, 'mv':lmv, 'ma':lma, 'b':lb, 'vA':lvA, 'v-a':v_a, 'l':lam, 'vc':lvc, '-vc':[-x for x in lvc], 'ac':lac, 'v-a-c':v_a_c}
	return do

def locus1(lam, t, x0, xm, s, l, d, n = 0, m = 10000):
	"""Locus starting from asymtotic solutions other than eigensolutions with varying parameters and fixed starting points."""
	lv = []
	la = []
	ls = []
	b = l[0]
	ld = []
	step = 0
	B = 1
	if l[1]/l[0] > 10*n:
		if n>0:
			B = math.pow(l[1]/l[0], 1.0/n)
		for i in range(n+1):
			step = B*b - b
			d[s] = b
			A = begin(lam, t, x0, d)
			do = A.to(xm, m)
			lv.append(do['v-a'][0])
			la.append(do['v-a'][1])
			ls.append(b)
			ld.append(do)
			b = b + step
	else:
		if n>0:
			step = (l[1]-l[0])/n
		for i in range(n+1):
			d[s] = b
			A = begin(lam, t, x0, d)
			do = A.to(xm, m)
			lv.append(do['v-a'][0])
			la.append(do['v-a'][1])
			ls.append(b)
			ld.append(do)
			b = b + step
	dl = {'v':lv, 'a':la, s:ls, 'm':m, 'n':n+1, 'd':ld}
	return dl

def locus3(lam, t, xm, l, d, n = 0, m = 10000):
	"""Locus starting from asymtotic solutions other than eigensolutions with fixed parameters and varying starting points."""
	lv = []
	la = []
	ls = []
	b = l[0]
	ld = []
	step = 0
	B = 1
	if l[1]/l[0] > 10*n:
		if n>0:
			B = math.pow(l[1]/l[0], 1.0/n)
		for i in range(n+1):
			step = B*b - b
			d['x'] = b
			A = begin(lam, t, b, d)
			do = A.to(xm, m)
			lv.append(do['v-a'][0])
			la.append(do['v-a'][1])
			ls.append(b)
			ld.append(do)
			b = b + step
	else:
		if n>0:
			step = (l[1]-l[0])/n
		for i in range(n+1):
			d['x'] = b
			A = begin(lam, t, b, d)
			do = A.to(xm, m)
			lv.append(do['v-a'][0])
			la.append(do['v-a'][1])
			ls.append(b)
			ld.append(do)
			b = b + step
	dl = {'v':lv, 'a':la, 'x':ls, 'm':m, 'n':n+1, 'd':ld}
	return dl
	
def locus2(lam, t, xm, l, d, c = 0, n = 0, mindx = 1.0/10**5, m0 = 10000):
	"""Locus starting from eigensolutions."""
	lv = []
	la = []
	ls = []
	ldx = []
	lm = []
	ld = []
	b = l[0]
	step = 0
	B = 1
	if l[1]/l[0] > 10*n:
		if n>0:
			B = math.pow(l[1]/l[0], 1.0/n)
		for i in range(n+1):
			m =m0
			step = B*b - b
			if abs(d['dx'])>b/100:
				dx_ = b/100
			else:
				dx_ = d['dx']
			dx = dx_
			A = begin(lam, t, b, {'dx':dx})
			if abs(xm-b)/m>abs(dx):
				if abs(dx)<mindx:
					m = int(abs(xm-b)/abs(mindx)+1000)
				else:		
					m = int(1000+abs(xm-b)/abs(dx))
			do = A.to(xm, m)
			if c==0:
				while (do['v-a'][0]<do['v-a-c'][0])and(abs(dx)<b/10):
					dx = dx*2
					A = begin(lam, t, b, {'dx':dx})
					do = A.to(xm, m)
				if do['v-a'][0]<do['v-a-c'][0]:
					dx = -dx_
					A = begin(lam, t, b, {'dx':dx})
					do = A.to(xm, m)
					while (do['v-a'][0]<do['v-a-c'][0])and(abs(dx)<b/10):
						dx = dx*2
						A = begin(lam, t, b, {'dx':dx})
						do = A.to(xm, m)
					if do['v-a'][0]<do['v-a-c'][0]:
						print(b)
						b = b + step
						continue
			else :
				while (do['v-a'][0]>do['v-a-c'][0])and(abs(dx)<b/10):
					dx = dx*2
					A = begin(lam, t, b, {'dx':dx})
					do = A.to(xm, m)
				if do['v-a'][0]>do['v-a-c'][0]:
					dx = -dx_
					A = begin(lam, t, b, {'dx':dx})
					do = A.to(xm, m)
					while (do['v-a'][0]>do['v-a-c'][0])and(abs(dx)<b/10):
						dx = dx*2
						A = begin(lam, t, b, {'dx':dx})
						do = A.to(xm, m)
					if do['v-a'][0]>do['v-a-c'][0]:
						print(b)
						b = b + step
						continue
			lv.append(do['v-a'][0])
			la.append(do['v-a'][1])
			ls.append(b)
			lm.append(m)
			ldx.append(dx)
			ld.append(do)
			b = b + step
	else:
		if n>0:
			step = (l[1]-l[0])/n
		for i in range(n+1):
			m =m0
			if abs(d['dx'])>b/100:
				dx_ = b/100
			else:
				dx_ = d['dx']
			dx = dx_
			A = begin(lam, t, b, {'dx':dx})
			if abs(xm-b)/m>abs(dx):
				if abs(dx)<mindx:
					m = int(abs(xm-b)/abs(mindx)+1000)
				else:		
					m = int(1000+abs(xm-b)/abs(dx))
			do = A.to(xm, m)
			if c==0:
				while (do['v-a'][0]<do['v-a-c'][0])and(abs(dx)<b/10):
					dx = dx*2
					A = begin(lam, t, b, {'dx':dx})
					do = A.to(xm, m)
				if do['v-a'][0]<do['v-a-c'][0]:
					dx = -dx_
					A = begin(lam, t, b, {'dx':dx})
					do = A.to(xm, m)
					while (do['v-a'][0]<do['v-a-c'][0])and(abs(dx)<b/10):
						dx = dx*2
						A = begin(lam, t, b, {'dx':dx})
						do = A.to(xm, m)
					if do['v-a'][0]<do['v-a-c'][0]:
						print(b)
						b = b + step
						continue
			else :
				while (do['v-a'][0]>do['v-a-c'][0])and(abs(dx)<b/10):
					dx = dx*2
					A = begin(lam, t, b, {'dx':dx})
					do = A.to(xm, m)
				if do['v-a'][0]>do['v-a-c'][0]:
					dx = -dx_
					A = begin(lam, t, b, {'dx':dx})
					do = A.to(xm, m)
					while (do['v-a'][0]>do['v-a-c'][0])and(abs(dx)<b/10):
						dx = dx*2
						A = begin(lam, t, b, {'dx':dx})
						do = A.to(xm, m)
					if do['v-a'][0]>do['v-a-c'][0]:
						print(b)
						b = b + step
						continue
			lv.append(do['v-a'][0])
			la.append(do['v-a'][1])
			ls.append(b)
			lm.append(m)
			ldx.append(dx)
			ld.append(do)
			b = b + step
	dl = {'v':lv, 'a':la, 'x':ls, 'dx':ldx, 'm':lm, 'n':n+1, 'd':ld}
	return dl

def locus_s1(lam, t, x0, xm, l, d = {}, h = 0, n = 0, mindx = 1.0/10**5, m0 = 10000):
	"""Locus with the one-temperature shock and varing xs."""
	step = 0
	if n>0:
		step = (l[1]-l[0])/n
	lv, la, ls, ldx, lm, ld= [], [], [], [], [], []
	lv_a_s1, lv_a_s2 = [], []
	b = l[0]
	B = 1
	if t==-1:
		for i in range(n+1):
			m = m0
			A = begin(lam, 0, 20.0, {'A':2.0, 'V':0.0})
			do = A.SIS(b)
			l = shock1(b, do['v-a'], lam ,h)
			if l==[]:
				print(b)
				b = b + step
				continue
			B = begin(lam, 4, b, {'v':l[1], 'a':l[2]})
			do_ = B.to(xm, m)
			lv.append(do_['v-a'][0])
			la.append(do_['v-a'][1])
			ls.append(b)
			lv_a_s1.append(do['v-a'])
			lv_a_s2.append([l[1],l[2]])
			if xm<b:
				ld.append(connect(do_,do,1))
			else:
				ld.append(connect(do,do_,2))
			b = b + step
		dl = {'v':lv, 'a':la, 'x':ls, 'm':m, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2 }
		return dl
	if (t==2)or(t==3):
		for i in range(n+1):
			m =m0
			c = d['c']
			if abs(d['dx'])>x0/100:
				dx = x0/100
			else:
				dx = d['dx']
			A = begin(lam, t, x0, {'dx':dx})
			if abs(x0-b)/m>abs(dx):
				if abs(dx)<mindx:
					m = int(abs(x0-b)/abs(mindx)+1000)
				else:		
					m = int(1000+abs(x0-b)/abs(dx))
			do = A.to(b, m)
			if c==0:
				if do['v-a'][0]<do['v-a-c'][0]:
					while (do['v-a'][0]<do['v-a-c'][0])and(abs(dx)<x0/10):
						dx = dx*2
						A = begin(lam, t, x0, {'dx':dx})
						do = A.to(b, m)
					if do['v-a'][0]<do['v-a-c'][0]:
						print(b)
						b = b + step
						continue
			else :
				if do['v-a'][0]>do['v-a-c'][0]:
					while (do['v-a'][0]>do['v-a-c'][0])and(abs(dx)<x0/10):
						dx = dx*2
						A = begin(lam, t, x0, {'dx':dx})
						do = A.to(b, m)
					if do['v-a'][0]>do['v-a-c'][0]:
						print(b)
						b = b + step
						continue
			l = shock1(b, do['v-a'], lam, h)
			if l==[]:
				print(b)
				b = b+step
				continue
			B = begin(lam, 4, b, {'v':l[1], 'a':l[2]})
			do_ = B.to(xm, m)
			lv.append(do_['v-a'][0])
			la.append(do_['v-a'][1])
			ls.append(b)
			ldx.append(dx)
			lm.append(m)
			b = b + step
			lv_a_s1.append(do['v-a'])
			lv_a_s2.append([l[1],l[2]])
			if xm<b:
				ld.append(connect(do_,do,1))
			else:
				ld.append(connect(do,do_,2))
		dl = {'v':lv, 'a':la, 'x':ls, 'dx':ldx, 'm':lm, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2}
		return dl
	for i in range(n+1):
		m =m0
		A = begin(lam, t, x0, d)
		do = A.to(b, m)
		l = shock1(b, do['v-a'], lam ,h)
		if l==[]:
			print(b)
			b = b + step
			continue
		B = begin(lam, 4, b, {'v':l[1], 'a':l[2]})
		do_ = B.to(xm, m)
		lv.append(do_['v-a'][0])
		la.append(do_['v-a'][1])
		ls.append(b)
		lv_a_s1.append(do['v-a'])
		lv_a_s2.append([l[1],l[2]])
		b = b + step
		if xm<b:
			ld.append(connect(do_,do,1))
		else:
			ld.append(connect(do,do_,2))
	dl = {'v':lv, 'a':la, 'x':ls, 'm':m, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2}
	return dl

def locus_s2(lam, t, x0, xm, l, d = {}, X = 1.0/1.5, n = 0, k = 0, f = 1, mindx = 1.0/10**5, m0 = 10000):
	"""Locus with the two-temperature shock and varying xsu/d."""
	step = 0
	if n>0:
		step = (l[1]-l[0])*X/n
	lv, la, ls, lm, ldx, ld = [], [], [], [], [], []
	lv_a_s1, lv_a_s2 = [], []
	b = l[0]*X
	B = 1
	if t==-1:
		for i in range(n+1):
			m =m0
			A = begin(lam, 0, 20.0, {'A':2.0, 'V':0.0})
			do = A.SIS(b)
			if k==0:
				l = shock2(b, do['v-a'], lam ,X)
			elif k==1:
				l = shock2_(b, do['v-a'], lam ,X)
			else: 
				l = shock2_(b, do['v-a'], lam, X, 2)
			if l==[]:
				print(b)
				b = b + step
				continue
			B = begin(lam, 4, b/X, {'v':l[1], 'a':l[2]})
			do_ = B.to(xm, m)
			lv.append(do_['v-a'][0])
			la.append(do_['v-a'][1])
			ls.append(b)
			lv_a_s1.append(do['v-a'])
			lv_a_s2.append([l[1],l[2]])
			b = b + step
			if f==1:
				ld.append(connect(do,do_,2))
			else:
				ld.append(connect(do_,do,1))
		dl = {'v':lv, 'a':la, 'x':ls, 'm':m, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2}
		return dl
	if (t==2)or(t==3):
		for i in range(n+1):
			m = m0
			c = d['c']
			if abs(d['dx'])>x0/100:
				dx = x0/100
			else:
				dx = d['dx']
			A = begin(lam, t, x0, {'dx':dx})
			if abs(x0-b)/m>abs(dx):
				if abs(dx)<mindx:
					m = int(abs(x0-b)/abs(mindx)+1000)
				else:		
					m = int(1000+abs(x0-b)/abs(dx))
			do = A.to(b, m)
			if c==0:
				if do['v-a'][0]<do['v-a-c'][0]:
					while (do['v-a'][0]<do['v-a-c'][0])and(abs(dx)<x0/10):
						dx = dx*2
						A = begin(lam, t, x0, {'dx':dx})
						do = A.to(b, m)
					if do['v-a'][0]<do['v-a-c'][0]:
						print(b)
						b = b + step
						continue
			else :
				if do['v-a'][0]>do['v-a-c'][0]:
					while (do['v-a'][0]>do['v-a-c'][0])and(abs(dx)<x0/10):
						dx = dx*2
						A = begin(lam, t, x0, {'dx':dx})
						do = A.to(b, m)
					if do['v-a'][0]>do['v-a-c'][0]:
						print(b)
						b = b + step
						continue
			if k==0:
				l = shock2(b, do['v-a'], lam ,X)
			elif k==1:
				l = shock2_(b, do['v-a'], lam ,X)
			else: 
				l = shock2_(b, do['v-a'], lam, X, 2)
			if l==[]:
				print(b)
				b = b+step
				continue
			B = begin(lam, 4, b/X, {'v':l[1], 'a':l[2]})
			do_ = B.to(xm, m)
			lv.append(do_['v-a'][0])
			la.append(do_['v-a'][1])
			ls.append(b)
			lm.append(m)
			ldx.append(dx)
			b = b + step
			lv_a_s1.append(do['v-a'])
			lv_a_s2.append([l[1],l[2]])
			if f==1:
				ld.append(connect(do,do_,2))
			else:
				ld.append(connect(do_,do,1))
		dl = {'v':lv, 'a':la, 'x':ls, 'm':lm, 'dx':ldx, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2}
		return dl
	for i in range(n+1):
		m = m0
		A = begin(lam, t, x0, d)
		do = A.to(b, m)
		if k==0:
			l = shock2(b, do['v-a'], lam ,X)
		elif k==1:
			l = shock2_(b, do['v-a'], lam ,X)
		else: 
			l = shock2_(b, do['v-a'], lam, X, 2)
		if l==[]:
			print(b)
			b = b + step
			continue
		B = begin(lam, 4, b/X, {'v':l[1], 'a':l[2]})
		do_ = B.to(xm, m)
		lv.append(do_['v-a'][0])
		la.append(do_['v-a'][1])
		ls.append(b)
		b = b + step
		lv_a_s1.append(do['v-a'])
		lv_a_s2.append([l[1],l[2]])
		if f==1:
			ld.append(connect(do,do_,2))
		else:
			ld.append(connect(do_,do,1))
	dl = {'v':lv, 'a':la, 'x':ls, 'm':m, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2 }
	return dl


def locus_s11(lam, t, xs, xm, l, d = {}, h = 0, n = 0, mindx = 1.0/10**5, m0 = 10000):
	"""Locus with the one-temperature shock and varying starting points."""
	lv, la, ls, lm, ldx, ld = [], [], [], [], [], []
	lv_a_s1, lv_a_s2 = [], []
	dl = {}
	b = l[0]
	step = 0
	B = 1
	if l[1]/l[0] > 10*n:
		if n>0:
			B = math.pow(l[1]/l[0], 1.0/n)
		if (t==2)or(t==3):
			for i in range(n+1):
				step = B*b-b
				l = locus_s1(lam, t, b, xm, [xs, 1], d, h, 0, mindx, m0)
				if l['v'] == []:
					b = b+step
					continue
				else:
					lv.append(l['v'][0])
					la.append(l['a'][0])
					ls.append(b)
					lm.append(l['m'][0])
					ldx.append(l['dx'][0])
					ld.append(l['d'][0])
					lv_a_s1.append(l['s1'][0])
					lv_a_s2.append(l['s2'][0])
					b = b+step
			dl = {'v':lv, 'a':la, 'x':ls, 'dx':ldx, 'm':lm, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2}
			return dl
		else:
			for i in range(n+1):
				step = B*b-b
				l = locus_s1(lam, t, b, xm, [xs, 1], d, h, 0, mindx, m0)
				if l['v'] == []:
					b = b+step
					continue
				else:
					lv.append(l['v'][0])
					la.append(l['a'][0])
					ls.append(b)
					m = m0
					ld.append(l['d'][0])
					lv_a_s1.append(l['s1'][0])
					lv_a_s2.append(l['s2'][0])
					b = b+step
			dl = {'v':lv, 'a':la, 'x':ls, 'm':m, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2 }
			return dl
	else:
		if n>0:
			step = (l[1]-l[0])/n
		if (t==2)or(t==3):
			for i in range(n+1):
				l = locus_s1(lam, t, b, xm, [xs, 1], d, h, 0, mindx, m0)
				if l['v'] == []:
					b = b+step
					continue
				else:
					lv.append(l['v'][0])
					la.append(l['a'][0])
					ls.append(b)
					lm.append(l['m'][0])
					ldx.append(l['dx'][0])
					ld.append(l['d'][0])
					lv_a_s1.append(l['s1'][0])
					lv_a_s2.append(l['s2'][0])
					b = b+step
			dl = {'v':lv, 'a':la, 'x':ls, 'dx':ldx, 'm':lm, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2}
			return dl
		else:
			for i in range(n+1):
				l = locus_s1(lam, t, b, xm, [xs, 1], d, h, 0, mindx, m0)
				if l['v'] == []:
					b = b+step
					continue
				else:
					lv.append(l['v'][0])
					la.append(l['a'][0])
					ls.append(b)
					m = m0
					ld.append(l['d'][0])
					lv_a_s1.append(l['s1'][0])
					lv_a_s2.append(l['s2'][0])
					b = b+step
			dl = {'v':lv, 'a':la, 'x':ls, 'm':m, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2 }
			return dl


def locus_s21(lam, t, xs, xm, l, d = {}, X = 1.0/1.5, n = 0, k = 0, f = 1, mindx = 1.0/10**5, m0 = 10000):
	"""Locus with the two-temperature shock and varying starting points."""
	lv, la, ls, lm, ldx, ld = [], [], [], [], [], []
	lv_a_s1, lv_a_s2 = [], []
	dl = {}
	b = l[0]
	step = 0
	B = 1
	if l[1]/l[0] > 10*n:
		if n>0:
			B = math.pow(l[1]/l[0], 1.0/n)
		if (t==2)or(t==3):
			for i in range(n+1):
				step = B*b-b
				l = locus_s2(lam, t, b, xm, [xs, 1], d, X, 0, k, f, mindx, m0)
				if l['v'] == []:
					b = b+step
					continue
				else:
					lv.append(l['v'][0])
					la.append(l['a'][0])
					ls.append(b)
					lm.append(l['m'][0])
					ldx.append(l['dx'][0])
					ld.append(l['d'][0])
					lv_a_s1.append(l['s1'][0])
					lv_a_s2.append(l['s2'][0])
					b = b+step
			dl = {'v':lv, 'a':la, 'x':ls, 'dx':ldx, 'm':lm, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2}
			return dl
		else:
			for i in range(n+1):
				step = B*b-b
				l = locus_s2(lam, t, b, xm, [xs, 1], d, X, 0, k, f, mindx, m0)
				if l['v'] == []:
					b = b+step
					continue
				else:
					lv.append(l['v'][0])
					la.append(l['a'][0])
					ls.append(b)
					m = m0
					ld.append(l['d'][0])
					lv_a_s1.append(l['s1'][0])
					lv_a_s2.append(l['s2'][0])
					b = b+step
			dl = {'v':lv, 'a':la, 'x':ls, 'm':m, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2 }
			return dl
	else:
		if n>0:
			step = (l[1]-l[0])/n
		if (t==2)or(t==3):
			for i in range(n+1):
				l = locus_s2(lam, t, b, xm, [xs, 1], d, X, 0, k, f, mindx, m0)
				if l['v'] == []:
					b = b+step
					continue
				else:
					lv.append(l['v'][0])
					la.append(l['a'][0])
					ls.append(b)
					lm.append(l['m'][0])
					ldx.append(l['dx'][0])
					ld.append(l['d'][0])
					lv_a_s1.append(l['s1'][0])
					lv_a_s2.append(l['s2'][0])
					b = b+step
			dl = {'v':lv, 'a':la, 'x':ls, 'dx':ldx, 'm':lm, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2}
			return dl
		else:
			for i in range(n+1):
				l = locus_s2(lam, t, b, xm, [xs, 1], d, X, 0, k, f, mindx, m0)
				if l['v'] == []:
					b = b+step
					continue
				else:
					lv.append(l['v'][0])
					la.append(l['a'][0])
					ls.append(b)
					m = m0
					ld.append(l['d'][0])
					lv_a_s1.append(l['s1'][0])
					lv_a_s2.append(l['s2'][0])
					b = b+step
			dl = {'v':lv, 'a':la, 'x':ls, 'm':m, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2 }
			return dl

def locus_s22(lam, t, x0, xs, xm, l, d = {}, n = 0, k = 0, f = 1, mindx = 1.0/10**5, m0 = 10000):
	"""Locus with the two-temperature shock and varying shock strength."""
	lv, la, ls, lm, ldx, ld = [], [], [], [], [], []
	lv_a_s1, lv_a_s2 = [], []
	dl = {}
	b = l[0]
	step = 0
	if n>0:
		step = (l[1]-l[0])/n
	if (t==2)or(t==3):
		for i in range(n+1):
			l = locus_s2(lam, t, x0, xm, [xs, 1], d, b, 0, k, f, mindx, m0)
			if l['v'] == []:
				b = b+step
				continue
			else:
				lv.append(l['v'][0])
				la.append(l['a'][0])
				ls.append(b)
				lm.append(l['m'][0])
				ldx.append(l['dx'][0])
				ld.append(l['d'][0])
				lv_a_s1.append(l['s1'][0])
				lv_a_s2.append(l['s2'][0])
				b = b+step
		dl = {'v':lv, 'a':la, 'x':ls, 'dx':ldx, 'm':lm, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2}
		return dl
	else:
		for i in range(n+1):
			l = locus_s2(lam, t, x0, xm, [xs, 1], d, b, 0, k, f, mindx, m0)
			if l['v'] == []:
				b = b+step
				continue
			else:
				lv.append(l['v'][0])
				la.append(l['a'][0])
				ls.append(b)
				m = m0
				ld.append(l['d'][0])
				lv_a_s1.append(l['s1'][0])
				lv_a_s2.append(l['s2'][0])
				b = b+step
		dl = {'v':lv, 'a':la, 'x':ls, 'm':m, 'n':len(ls), 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2 }
		return dl

def connect(d1,d2,k = 0):
	d = {}
	d['vc'] = d1['vc']+d2['vc']
	d['ac'] = d1['ac']+d2['ac']
	d['-vc'] = d1['-vc']+d2['-vc']
	d['n'] = d1['n']+d2['n']
	d['x'] = d1['x']+d2['x']
	d['a'] = d1['a']+d2['a']
	d['v'] = d1['v']+d2['v']
	d['-v']= d1['-v']+d2['-v']
	d['vA'] = d1['vA']+d2['vA']
	d['m'] = d1['m']+d2['m']
	d['b'] = d1['b']+d2['b']
	d['l'] = d1['l']
	if k==1:
		d['v-a'] = d1['v-a']
		d['v-a-c'] = d1['v-a-c']
	elif k==2:
		d['v-a'] = d2['v-a']
		d['v-a-c'] = d2['v-a-c']
	else:
		if d1['v-a']==[d1['v'][0],d1['a'][0]]:
			d['v-a'] = d1['v-a']
		else:
			d['v-a'] = d2['v-a']
		if d1['v-a-c']==[d1['vc'][0],d1['ac'][0]]:
			d['v-a-c'] = d1['v-a-c']
		else:
			d['v-a-c'] = d2['v-a-c']
	return d
	
def void(d2):
	d = {}
	d['vc'] = [d2['vc'][0],d2['vc'][0]]+d2['vc']
	d['ac'] = [d2['ac'][0],d2['ac'][0]]+d2['ac']
	d['-vc'] = [d2['-vc'][0],d2['-vc'][0]]+d2['-vc']
	d['n'] = 2+d2['n']
	d['x'] = [0,d2['x'][0]]+d2['x']
	d['a'] = [0,0]+d2['a']
	d['v'] = [0,0]+d2['v']
	d['-v']= [0,0]+d2['-v']
	d['vA'] = [0,0]+d2['vA']
	d['m'] = [0,0]+d2['m']
	d['b'] = [0,0]+d2['b']
	d['l'] = d2['l']
	d['v-a'] = d2['v-a']
	d['v-a-c'] = d2['v-a-c']
	return d

		
def R(d, D):
	l = [d['x'][x]**2*d['a'][x]/D for x in range(d['n'])]
	return l

def Es(d, n, k = 0):
	do = {}
	lv, la, ls, lm, ld, lv_a_s1, lv_a_s2, ldx = [], [], [], [], [], [], [], []
	if k==0:
		for i in range(d['n']):
			if i == n:
				print(d['x'][i])
			else:
				lv.append(d['v'][i])
				la.append(d['a'][i])
				ls.append(d['x'][i])
				ld.append(d['d'][i])
				lv_a_s1.append(d['s1'][i])
				lv_a_s2.append(d['s2'][i])
		do = {'v':lv, 'a':la, 'x':ls, 'm':l['m'], 'n':d['n']-1, 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2}
		return do
	else: 
		for i in range(d['n']):
			if i == n:
				print(d['x'][i])
			else:
				lv.append(d['v'][i])
				la.append(d['a'][i])
				ls.append(d['x'][i])
				lm.append(d['m'][i])
				ldx.append(d['dx'][i])
				ld.append(d['d'][i])
				lv_a_s1.append(d['s1'][i])
				lv_a_s2.append(d['s2'][i])
		do = {'v':lv, 'a':la, 'x':ls, 'm':lm, 'dx':ldx, 'n':d['n']-1, 'd':ld, 's1':lv_a_s1, 's2':lv_a_s2}
		return do
		
def E1(d, n, s):
	dl = {}
	lv, la, ls, ld = [], [], [], []
	for i in range(d['n']):
		if i == n:
			print(d['x'][i])
		else:
			lv.append(d['v'][i])
			la.append(d['a'][i])
			ls.append(d[s][i])
			ld.append(d['d'][i])
	dl = {'v':lv, 'a':la, s:ls, 'm':d['m'], 'n':d['n']-1, 'd':ld}
	return dl



def E2(d, n):
	dl = {}
	lv, la, ls, lm, ld, ldx = [], [], [], [], [], []
	for i in range(d['n']):
		if i == n:
			print(d['x'][i])
		else:
			lv.append(d['v'][i])
			la.append(d['a'][i])
			ls.append(d['x'][i])
			lm.append(d['m'][i])
			ldx.append(d['dx'][i])
			ld.append(d['d'][i])
	dl = {'v':lv, 'a':la, 'x':ls, 'dx':ldx, 'm':lm, 'n':d['n']-1, 'd':ld}
	return dl

def VA(d):
	A = d['a'][d['n']-1]*d['x'][d['n']-1]**2
	V = d['v'][d['n']-1]
	X = d['x'][d['n']-1]
	l = [X,V,A]
	return l

