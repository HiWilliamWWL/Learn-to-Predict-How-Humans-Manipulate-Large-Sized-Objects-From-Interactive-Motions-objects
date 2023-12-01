import math
class ParticlesControl:
    def __init__(self, unit = 0.15, floorHeight = -10):
        self.particles = []
        self.constraints = []
        self.delta_t_set = 0.001
        self.NUM_ITER1 = 5
        self.unit = unit
        self.pos = [0, 0, 0]
        self.mouse = False
        self.floorHeight = floorHeight
    
    def addParticle(self, x, y, z, m=1.0):
        self.particles.append(Particle(x*self.unit, y*self.unit, z*self.unit, m, self, self.floorHeight))
    
    def addConstraint(self, index0, index1, visibale = False, fixBound = False):
        self.constraints.append(Constraint(index0, index1, visibale, fixBound, self.particles))
    
    def update(self):
        for i in range(self.NUM_ITER1):
            for p in self.particles:
                p.update(self.delta_t_set)
            for c in self.constraints:
                c.update()



class Particle:
    ay = -9.8
    ax = 0.0
    az = 0.0

    def __init__(self, x, y, z, m, superControl, floorHeight):
        self.m = 1.0
        self.x = x * 1.0
        self.y = y * 1.0
        self.z = z * 1.0
        '''
        self.oldx = x
        self.oldy = y
        self.oldz = z
        '''
        self.accx = 0.0
        self.accy = 0.0
        self.accz = 0.0
        self.newx = x
        self.newy = y
        self.newz = z
        self.superControl = superControl
        self.ax1 = 0.0
        self.ay1 = 0.0
        self.az1 = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        
        self.fixed = False
        self.selected = False
        self.floorHeight = floorHeight
        
    def apply_forces(self):
        theDrag = 0.08
        totalFx, totalFy, totalFz = (self.ax+self.ax1, self.ay+self.ay1, self.az+self.az1)
        dragFx, dragFy, dragFz = (0.5 * theDrag * self.vx * abs(self.vx), 0.5 * theDrag * self.vy * abs(self.vy), 0.5 * theDrag *self.vz * abs(self.vz))
        dragAccx, dragAccy, dragAccz =(dragFx / self.m, dragFy / self.m, dragFz / self.m)
        return (totalFx - dragAccx, totalFy - dragAccy, totalFz -dragAccz)

    def update(self, delta_t):
        if self.fixed == False:
            self.newx = self.x + self.vx*delta_t + self.accx *(delta_t*delta_t*0.5)
            self.newy = self.y + self.vy*delta_t + self.accy *(delta_t*delta_t*0.5)
            self.newz = self.z + self.vz*delta_t + self.accz *(delta_t*delta_t*0.5)
            oldAccx, oldAccy, oldAccz = (self.accx, self.accy, self.accz)
            self.accx, self.accy, self.accz = self.apply_forces()
            self.vx = self.vx + (oldAccx + self.accx) * (delta_t * 0.5)
            self.vy = self.vy + (oldAccy + self.accy) * (delta_t * 0.5)
            self.vz = self.vz + (oldAccz + self.accz) * (delta_t * 0.5)
            self.x, self.y, self.z = (self.newx, self.newy, self.newz)

            if self.y < self.floorHeight * 0.15 - 0.01 or self.y > 10:
                #self.y, self.oldy = self.oldy, self.y
                #self.y, self.oldy = self.oldy, self.y
                self.vy = -self.vy
                self.accy = -self.accy
                self.fixed = True
                #self.ay = 0.0
                #self.z, self.oldz = self.oldz, self.oldz
                
        if self.selected == True:
            #pos = pygame.mouse.get_pos()
            self.x += self.superControl.pos[0]
            self.y += self.superControl.pos[1]
            self.z += self.superControl.pos[2]
        if self.superControl.mouse == False:
            self.selected = False
        
class Constraint(ParticlesControl):
    def __init__(self, index0, index1, visibale, fixBound, particles):
        self.particles = particles
        self.index0 = index0
        self.index1 = index1
        self.visibale = visibale
        self.fixBound = fixBound
        delta_x = self.particles[index0].x * 1.0 - self.particles[index1].x* 1.0
        delta_y = self.particles[index0].y * 1.0- self.particles[index1].y* 1.0
        delta_z = self.particles[index0].z * 1.0- self.particles[index1].z* 1.0
        self.restLength = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
        
    def update(self):
        if (self.fixBound):
            delta_x = self.particles[self.index0].x * 1.0 - self.particles[self.index0].oldx* 1.0
            delta_y = self.particles[self.index0].y * 1.0 - self.particles[self.index0].oldy* 1.0
            delta_z = self.particles[self.index0].z * 1.0 - self.particles[self.index0].oldz* 1.0
            self.particles[self.index1].x = self.particles[self.index1].oldx + delta_x
            self.particles[self.index1].y = self.particles[self.index1].oldy + delta_y
            self.particles[self.index1].z = self.particles[self.index1].oldz + delta_z
            return

        delta_x = self.particles[self.index1].x * 1.0 - self.particles[self.index0].x* 1.0
        delta_y = self.particles[self.index1].y * 1.0 - self.particles[self.index0].y* 1.0
        delta_z = self.particles[self.index1].z * 1.0 - self.particles[self.index0].z* 1.0
        deltaLength = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
        diff = (deltaLength - self.restLength)/deltaLength

        self.particles[self.index0].x += 0.5 * diff * delta_x
        self.particles[self.index0].z += 0.5 * diff * delta_z
        if self.particles[self.index0].fixed == False:
            self.particles[self.index0].y += 0.5 * diff * delta_y

        self.particles[self.index1].x -= 0.5 * diff * delta_x
        self.particles[self.index1].z -= 0.5 * diff * delta_z 
        if self.particles[self.index1].fixed == False:
            self.particles[self.index1].y -= 0.5 * diff * delta_y