
!##########################################################################
!###########################################################################
subroutine readinput
use velocity
use phase
use param
use surfactant
use mpivar
use particles
implicit none

open(unit=55,file='input.inp',form='formatted',status='old')
!Time step parameters
read(55,*) restart
read(55,*) tstart
read(55,*) tfin
read(55,*) dump
!Flow parameters
read(55,*) inflow
read(55,*) inphi
read(55,*) insurf
read(55,*) dt
read(55,*) muc
read(55,*) mud
read(55,*) rhoc
read(55,*) rhod
! forcing parameters
read(55,*) f1
read(55,*) f2
read(55,*) f3
read(55,*) k0
! phase-field parameters
read(55,*) radius
read(55,*) sigma
read(55,*) epsr   
! surfactant parameters
read(55,*) diffs
read(55,*) betas
 

! compute pre-defined constant 
twopi=8.0_8*atan(1.0_8)
lx=twopi
! dx = lx/(nx-1)
dx = lx/(nx)

dxi=1.d0/dx
ddxi=1.d0/dx/dx
rhoi=1.d0/rhoc
! reference density for the pressure splitting: must be the minimum density for stability
rho0=min(rhoc,rhod)
rho0i=1.d0/rho0
eps=epsr*dx
epsi=1.d0/eps
enum=1.e-16

if (rank .eq. 0) then
    !enable/disable for debug check parameters
    write(*,*) "----------------------------------------------"
    write(*,*) "███    ███ ██   ██ ██ ████████ ██████   ██████"  
    write(*,*) "████  ████ ██   ██ ██    ██         ██ ██"       
    write(*,*) "██ ████ ██ ███████ ██    ██     █████  ███████"  
    write(*,*) "██  ██  ██ ██   ██ ██    ██         ██ ██    ██" 
    write(*,*) "██      ██ ██   ██ ██    ██    ██████   ██████"         
    write(*,*) "----------------------------------------------"
    write(*,*) "----Homogeneous isotropic turbulence setup----"
    write(*,*) 'Grid:', nx, 'x', nx, 'x', nx
    write(*,*) "Restart ", restart
    write(*,*) "Tstart  ", tstart
    write(*,*) "Tfin    ", tfin
    write(*,*) "Dump    ", dump
    write(*,*) "Inflow  ", inflow
    write(*,*) "Deltat  ", dt
    write(*,*) "Mu carrier     ", muc
    write(*,*) "Mu drop        ", mud
    write(*,*) "Rho carrier    ", rhoc
    write(*,*) "Rho drop       ", rhod
    write(*,*) "Rho0 (ref.)    ", rho0
    write(*,*) "f1,f2,f3,k0", f1,f2,f3,k0
    write(*,*) "Radius  ", radius
    write(*,*) "Sigma   ", sigma
    write(*,*) "Eps     ", eps
    write(*,*) "diffsurf     ", diffs
    write(*,*) "Epsi    ", epsi
    write(*,*) "Lx      ", lx
    write(*,*) "dx", dx
    write(*,*) "dxi", dxi
    write(*,*) "ddxi", ddxi
    write(*,*) "rhoi", rhoi
endif

end subroutine


