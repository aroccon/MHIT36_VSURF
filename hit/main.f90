#define CHECK_CUDECOMP_EXIT(f) if (f /= CUDECOMP_RESULT_SUCCESS) call exit(1)
! Consistent mass-momentum transport (phase-field only):
! mass flux at the u,v,w faces = rhoc*u^n + (rhod-rhoc)*(time-integrated phi flux of the RK4 stages)
#define MFX(a,b,c) (rhoc*u(a,b,c) + drho*fbx(a,b,c))
#define MFY(a,b,c) (rhoc*v(a,b,c) + drho*fby(a,b,c))
#define MFZ(a,b,c) (rhoc*w(a,b,c) + drho*fbz(a,b,c))
! transported velocity extrapolated at t^{n+1/2}
#define UE(a,b,c) (1.5d0*u(a,b,c) - 0.5d0*uold(a,b,c))
#define VE(a,b,c) (1.5d0*v(a,b,c) - 0.5d0*vold(a,b,c))
#define WE(a,b,c) (1.5d0*w(a,b,c) - 0.5d0*wold(a,b,c))

program main
use cudafor
use cudecomp
use cufft
use mpi
use velocity 
use phase
use surfactant
use particles
use param
use mpivar
use cudecompvar

use, intrinsic :: ieee_arithmetic

implicit none
! timer for scaling test
real :: t_start, t_end, elapsed
! grid dimensions
integer :: comm_backend
integer :: pr, pc
! cudecomp
integer :: planXf, planXb
integer :: planY, planZ
integer :: batchsize
integer :: status
integer :: i,j,k,il,jl,kl,ig,jg,kg,t
integer :: im,ip,jm,jp,km,kp,last
integer :: inY,enY,inZ,enZ
!beginDEBUG
integer, parameter :: Mx = 1, My = 2, Mz = 1
!endDEBUG
real(8), device, allocatable :: kx_d(:)
! working arrays
complex(8), allocatable :: psi(:)
real(8), allocatable :: ua(:,:,:)
real(8), allocatable :: uaa(:,:,:)
real(8), allocatable :: psi_real(:)
! real(8), device, allocatable :: psi_real_d(:)
complex(8), device, allocatable :: psi_d(:)
complex(8), pointer, device, contiguous :: work_d(:), work_halo_d(:), work_d_d2z(:), work_halo_d_d2z(:)
character(len=40) :: namefile
character(len=4) :: itcount
! Code variables
real(8)::err,maxErr
complex(8), device, pointer :: phi3d(:,:,:)
real(8) :: k2
! variable density: inverse density at the u,v,w faces, extrapolated pressure, phi fluxes, momentum fluxes
real(8) :: rhoxi, rhoyi, rhozi
real(8) :: phat_c, phat_x, phat_y, phat_z
real(8) :: flx, fly, flz
real(8) :: ge, gw, gn, gs, gt, gb, rhon, rhonp
!integer :: il, jl, ig, jg
integer :: offsets(3), xoff, yoff
integer :: np(3)
! Enable or disable phase field 
#define phiflag 1
! Enable or disable surfactant (phase-field must be enabled)
#define surflag 1



!########################################################################################################################################
! 1. INITIALIZATION OF MPI AND cuDECOMP AUTOTUNING : START
!########################################################################################################################################
! safety check
#if surflag == 1
#if phiflag == 0
stop "Phase-field must be enabled to use surfactant; set phiflag to 1 or surflag to 0"
#endif
#endif
! MPI initialization, put in rank the local MPI rank number and ranks total number
! Same procedura defined in the cuDecomp documentation
call mpi_init(ierr)
call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
call mpi_comm_size(MPI_COMM_WORLD, ranks, ierr)

call mpi_comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, localComm, ierr)
call mpi_comm_rank(localComm, localRank, ierr)
ierr = cudaSetDevice(localRank) !assign GPU to MPI rank

! Define grid and decomposition
call readinput
drho=rhod-rhoc
#if phiflag == 0
! single phase: no density contrast, use the carrier density as reference
rho0=rhoc
rho0i=1.d0/rhoc
#endif

! number of process along each direction (r means y and c means z)
pr = 0
pc = 0
halo_ext=1
! comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_P2P
comm_backend = 0 ! Enable full autotuning

CHECK_CUDECOMP_EXIT(cudecompInit(handle, MPI_COMM_WORLD))

! config is a struct and pr and pc are the number of pencils along the two directions
! gdims is the global grid
! create an uninitialized configuration struct and initialize it to defaults using cudecompGridDescConfigSetDefaults. 
! Initializing to default values is required to ensure no entries are left uninitialized.
CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(config))
pdims = [pr, pc] !pr and pc are the number of pencil along the different directions
config%pdims = pdims
! gdims = [nx, ny, nz]
! config%gdims = gdims
halo = [0, halo_ext, halo_ext] ! no halo along x neeed because is periodic and in physical space i have x-pencil
! for transpositions
config%transpose_comm_backend = comm_backend
config%transpose_axis_contiguous = .true.
! for halo exchanges
config%halo_comm_backend = CUDECOMP_HALO_COMM_MPI
! Setting for periodic halos in all directions (non required to be in config)
halo_periods = [.true., .true., .true.]

! create spectral grid descriptor first to select pdims for optimal transposes
gdims = [nx/2+1, ny, nz]
config%gdims = gdims

! Set up autotuning options for spectral grid (transpose related settings)
CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(options))
options%dtype = CUDECOMP_DOUBLE_COMPLEX
if (comm_backend == 0) then
   options%autotune_transpose_backend = .true.
   options%autotune_halo_backend = .false.
endif
options%transpose_use_inplace_buffers = .true.
options%transpose_input_halo_extents(:, 1) = halo
options%transpose_output_halo_extents(:, 4) = halo

CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, grid_descD2Z, config, options))

! create physical grid descriptor
! take previous config and modify the global grid (nx instead of nx/2+1)
! reset transpose_comm_backend to default value to avoid picking up possible nvshmem
! transpose backend selection (this impacts how workspaces are allocated)
gdims = [nx, ny, nz]
config%gdims = gdims
config%transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_P2P

! Set up autotuning options for physical grid (halo related settings)
CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(options))
options%dtype = CUDECOMP_DOUBLE_COMPLEX
if (comm_backend == 0) then
   options%autotune_halo_backend = .true.
endif
options%halo_extents(:) = halo
options%halo_periods(:) = halo_periods
options%halo_axis = 1
CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, grid_desc, config, options))

!Print information on configuration
! issue with NVHPC > 25.X, simply avoid this call (not critical)
!if (rank == 0) then
!   write(*,"(' Running on ', i0, ' x ', i0, ' process grid ...')") config%pdims(1), config%pdims(2)
!   write(*,"(' Using ', a, ' transpose backend ...')") &
!         cudecompTransposeCommBackendToString(config%transpose_comm_backend)
!   write(*,"(' Using ', a, ' halo backend ...')") &
!         cudecompHaloCommBackendToString(config%halo_comm_backend)
!endif

! Get pencil info for the grid descriptor in the physical space
! This function returns a pencil struct (piX, piY or piZ) that contains the shape, global lower and upper index bounds (lo and hi), 
! size of the pencil, and an order array to indicate the memory layout that will be used (to handle permuted, axis-contiguous layouts).
! Additionally, there is a halo_extents data member that indicates the depth of halos for the pencil, by axis.
! If no halo regions are necessary, a NULL pointer can be provided in place of this array (or omitted)
! Pencil info in x-configuration present in PiX (shape,lo,hi,halo_extents,size)
CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piX, 1, halo))
nElemX = piX%size !<- number of total elments in x-configuratiion (including halo)
! Pencil info in Y-configuration present in PiY
CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piY, 2))
nElemY = piY%size
! Pencil info in Z-configuration present in PiZ
CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, piZ, 3))
nElemZ = piZ%size

! Get workspace sizes for transpose (1st row, not used) and halo (2nd row, used)
CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, nElemWork))
CHECK_CUDECOMP_EXIT(cudecompGetHaloWorkspaceSize(handle, grid_desc, 1, halo, nElemWork_halo))

CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_descD2Z, piX_d2z, 1,halo))
nElemX_d2z = piX_d2z%size !<- number of total elments in x-configuratiion (include halo)
! Pencil info in Y-configuration present in PiY
CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_descD2Z, piY_d2z, 2))
nElemY_d2z = piY_d2z%size
! Pencil info in Z-configuration present in PiZ
CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_descD2Z, piZ_d2z, 3))
nElemZ_d2z = piZ_d2z%size
! Get workspace sizes for transpose (1st row,used) and halo (2nd row, not used)
CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_descD2Z, nElemWork_d2z))
CHECK_CUDECOMP_EXIT(cudecompGetHaloWorkspaceSize(handle, grid_descD2Z, 1, halo, nElemWork_halo_d2z))


! CUFFT initialization -- Create Plans
! Forward 1D FFT in X: D2Z
batchSize = piX_d2z%shape(2)*piX_d2z%shape(3) !<- number of FFT (from x-pencil dimension)
status = cufftPlan1D(planXf, nx, CUFFT_D2Z, batchSize)
if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating X plan Forward'

! Backward 1D FFT in X: Z2D
batchSize = piX_d2z%shape(2)*piX_d2z%shape(3) !<- number of FFT (from x-pencil dimension)
status = cufftPlan1D(planXb, nx, CUFFT_Z2D, batchSize)
if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating X plan Backward'

! it's always 2 and 3 because y-pencil have coordinates y,z,x
batchSize = piY_d2z%shape(2)*piY_d2z%shape(3)
status = cufftPlan1D(planY, ny, CUFFT_Z2Z, batchSize)
if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Y plan Forward & Backward'

! it's always 2 and 3 because y-pencil have coordinates z,y,x
batchSize = piZ_d2z%shape(2)*piZ_d2z%shape(3)
status = cufftPlan1D(planZ, nz, CUFFT_Z2Z, batchSize)
if (status /= CUFFT_SUCCESS) write(*,*) rank, ': Error in creating Z plan Forward & Backward'



! define grid
allocate(x(nx),x_ext(nx+1))
x(1)= 0
do i = 2, nx
   x(i) = x(i-1) + dx
enddo

! define wavenumbers
allocate(kx(nx))
do i = 1, nx/2
   kx(i) = (i-1)*(twoPi/lx)
enddo
do i = nx/2+1, nx
   kx(i) = (i-1-nx)*(twoPi/LX)
enddo
! allocate k_d on the device (later on remove and use OpenACC + managed memory?)
allocate(kx_d, source=kx)

allocate(mysin(nx), mycos(nx))
do i=1,nx
   ! compute here the sin to avoid multiple computations of sin
   mysin(i)=sin(k0*x(i)+dx/2)
   ! compute here the cos to avoid multiple computations of cos
   mycos(i)=cos(k0*x(i)+dx/2)
enddo

!########################################################################################################################################
! 1. INITIALIZATION AND cuDECOMP AUTOTUNING : END
!########################################################################################################################################



!########################################################################################################################################
! START STEP 2: ALLOCATE ARRAYS
!########################################################################################################################################

! allocate arrays
allocate(psi(max(nElemX, nElemY, nElemZ))) !largest among the pencil
allocate(psi_real(max(nElemX, nElemY, nElemZ))) !largest among the pencil
allocate(psi_d(max(nElemX_d2z, nElemY_d2z, nElemZ_d2z))) ! phi on device
allocate(ua(nx, piX%shape(2), piX%shape(3)))

! Pressure variable
allocate(rhsp(piX%shape(1), piX%shape(2), piX%shape(3))) 
allocate(p(piX%shape(1), piX%shape(2), piX%shape(3)))
allocate(pold(piX%shape(1), piX%shape(2), piX%shape(3)))
! pressure history used by the Dodd & Ferrante splitting (zero at start)
p=0.d0
pold=0.d0

!allocate variables
!NS variables
allocate(u(piX%shape(1),piX%shape(2),piX%shape(3)),v(piX%shape(1),piX%shape(2),piX%shape(3)),w(piX%shape(1),piX%shape(2),piX%shape(3))) !velocity vector
! allocate(ustar(piX%shape(1),piX%shape(2),piX%shape(3)),vstar(piX%shape(1),piX%shape(2),piX%shape(3)),wstar(piX%shape(1),piX%shape(2),piX%shape(3))) ! provisional velocity field
allocate(rhsu(piX%shape(1),piX%shape(2),piX%shape(3)),rhsv(piX%shape(1),piX%shape(2),piX%shape(3)),rhsw(piX%shape(1),piX%shape(2),piX%shape(3))) ! right hand side u,v,w
allocate(rhsu_o(piX%shape(1),piX%shape(2),piX%shape(3)),rhsv_o(piX%shape(1),piX%shape(2),piX%shape(3)),rhsw_o(piX%shape(1),piX%shape(2),piX%shape(3))) ! right hand side u,v,w
allocate(mu(piX%shape(1),piX%shape(2),piX%shape(3)))
allocate(tau12(piX%shape(1),piX%shape(2),piX%shape(3)),tau13(piX%shape(1),piX%shape(2),piX%shape(3)),tau23(piX%shape(1),piX%shape(2),piX%shape(3)))
!PFM variables
#if phiflag == 1
allocate(phi(piX%shape(1),piX%shape(2),piX%shape(3)),q_phi(piX%shape(1),piX%shape(2),piX%shape(3)),rhsphi(piX%shape(1),piX%shape(2),piX%shape(3)),psidi(piX%shape(1),piX%shape(2),piX%shape(3)))
allocate(normx(piX%shape(1),piX%shape(2),piX%shape(3)),normy(piX%shape(1),piX%shape(2),piX%shape(3)),normz(piX%shape(1),piX%shape(2),piX%shape(3)))
allocate(fxst(piX%shape(1),piX%shape(2),piX%shape(3)),fyst(piX%shape(1),piX%shape(2),piX%shape(3)),fzst(piX%shape(1),piX%shape(2),piX%shape(3))) ! surface tension forces
! consistent mass-momentum transport
allocate(phio(piX%shape(1),piX%shape(2),piX%shape(3)))
allocate(qfx(piX%shape(1),piX%shape(2),piX%shape(3)),qfy(piX%shape(1),piX%shape(2),piX%shape(3)),qfz(piX%shape(1),piX%shape(2),piX%shape(3)))
allocate(fbx(piX%shape(1),piX%shape(2),piX%shape(3)),fby(piX%shape(1),piX%shape(2),piX%shape(3)),fbz(piX%shape(1),piX%shape(2),piX%shape(3)))
allocate(uold(piX%shape(1),piX%shape(2),piX%shape(3)),vold(piX%shape(1),piX%shape(2),piX%shape(3)),wold(piX%shape(1),piX%shape(2),piX%shape(3)))
#endif
!SURF FLAG
# if surflag == 1
allocate(surf(piX%shape(1),piX%shape(2),piX%shape(3)),rhsurf(piX%shape(1),piX%shape(2),piX%shape(3)),q_surf(piX%shape(1),piX%shape(2),piX%shape(3)))
#endif

! allocate arrays for transpositions and halo exchanges 
CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, work_d, nElemWork))
CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, work_halo_d, nElemWork_halo))
! allocate arrays for transpositions
CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_descD2Z, work_d_d2z, nElemWork_d2z))
CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_descD2Z, work_halo_d_d2z, nElemWork_halo_d2z))
!########################################################################################################################################
! END STEP2: ALLOCATE ARRAYS
!########################################################################################################################################



!########################################################################################################################################
! START STEP 3: FLOW, PHASE FIELD AND PARTICLES INIT
!########################################################################################################################################
! 3.1 Read/initialize from data without halo grid points (avoid out-of-bound if reading usin MPI I/O)
! 3.2 Call halo exchnages along Y and Z for u,v,w and phi
if (restart .eq. 0) then !fresh start Taylor Green or read from file in init folder
   if (rank.eq.0) write(*,*) "Initialize velocity field (fresh start)"
   if (inflow .eq. 0) then
      if (rank.eq.0) write(*,*) "Initialize Taylor-green"
      do k = 1+halo_ext, piX%shape(3)-halo_ext
         kg = piX%lo(3) + k - 1 
         do j = 1+halo_ext, piX%shape(2)-halo_ext
            jg = piX%lo(2) + j - 1 
            do i = 1, piX%shape(1)
               u(i,j,k) =   sin(x(i)-dx/2)*cos(x(jg))*cos(x(kg))
               v(i,j,k) =  -cos(x(i))*sin(x(jg)-dx/2)*cos(x(kg))
               w(i,j,k) =  0.d0
            enddo
         enddo
      enddo
   endif
   if (inflow .eq. 1) then
      if (rank.eq.0)  write(*,*) "Initialize from data"
      call readfield(1)
      call readfield(2)
      call readfield(3)
   endif
endif
if (restart .eq. 1) then !restart, ignore inflow and read the tstart field 
   if (rank.eq.0)  write(*,*) "Initialize velocity field (from output folder), iteration:", tstart
   call readfield_restart(tstart,1)
   call readfield_restart(tstart,2)
   call readfield_restart(tstart,3)
endif

! update halo cells along y and z directions (enough only if pr and pc are non-unitary)
!$acc host_data use_device(u,v,w)
CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, u, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, u, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, v, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, v, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, w, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, w, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
!$acc end host_data 

! find local maximum velocity (for ACDI)
uc=maxval(u)
vc=maxval(v)
wc=maxval(w)
umax=max(wc,max(uc,vc))
call MPI_Allreduce(umax,gumax,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD, ierr)

! initialize phase-field
#if phiflag == 1
   if (restart .eq. 0) then
      if (rank.eq.0) write(*,*) 'Initialize phase field (fresh start)'
      if (inphi .eq. 0) then
         if (rank.eq.0) write(*,*) 'Spherical drop'
         do k = 1+halo_ext, piX%shape(3)-halo_ext
            kg = piX%lo(3) + k - 1 
            do j = 1+halo_ext, piX%shape(2)-halo_ext
               jg = piX%lo(2) + j - 1 
               do i = 1, piX%shape(1)
                  pos=(x(i)-lx/2)**2d0 +  (x(jg)-lx/2)**2d0 + (x(kg)-lx/2)**2d0
                  phi(i,j,k) = 0.5d0*(1.d0-tanh((sqrt(pos)-radius)/2/eps))
               enddo
            enddo
         enddo
      endif
      if (inphi .eq. 1) then
         if (rank.eq.0)  write(*,*) "Initialize phase-field from data"
         call readfield(5)
      endif
   endif
   if (restart .eq. 1) then
      if (rank == 0) write(*,*) "Initialize phase-field (restart, from output folder), iteration:", tstart
      call readfield_restart(tstart,5)
   endif
   ! update halo
   !$acc host_data use_device(phi)
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, phi, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, phi, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   !$acc end host_data 
#endif

! initialize surfactant
#if surflag == 1
   if (restart .eq. 0) then
      if (rank.eq.0) write(*,*) 'Initialize surfactant'
      if (insurf .eq. 0) then
         if (rank.eq.0) write(*,*) 'Gradient of phi'
         do k = 1+halo_ext, piX%shape(3)-halo_ext
            do j = 1+halo_ext, piX%shape(2)-halo_ext
               do i = 1, piX%shape(1)
                  ip=i+1 
                  im=i-1
                  jp=j+1 
                  jm=j-1
                  kp=k+1 
                  km=k-1
                  if (ip > nx) ip=1
                  if (im < 1)  im=nx
                  val =       (0.5d0*(phi(ip,j,k)-phi(im,j,k))*dxi)**2
                  val = val + (0.5d0*(phi(i,jp,k)-phi(i,jm,k))*dxi)**2
                  val = val + (0.5d0*(phi(i,j,kp)-phi(i,j,km))*dxi)**2
                  surf(i,j,k) = sqrt(val)
               enddo
            enddo
         enddo
      endif
      if (inphi .eq. 1) then
         if (rank.eq.0)  write(*,*) "Initialize surfactant from data"
         call readfield(6)
      endif
   endif
   if (restart .eq. 1) then
      if (rank == 0) write(*,*) "Initialize surfactant (restart, from output folder), iteration:", tstart
      call readfield_restart(tstart,6)
   endif
   ! update halo
   !$acc host_data use_device(surf)
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, surf, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, surf, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   !$acc end host_data 
#endif

!Save initial fields (only if a fresh start)
if (restart .eq. 0) then
   if (rank.eq.0) write(*,*) "Save initial fields"
   call writefield(tstart,1)
   call writefield(tstart,2)
   call writefield(tstart,3)
   #if phiflag == 1
   call writefield(tstart,5)
   #endif
   #if surflag == 1
   call writefield(tstart,6)
   #endif
endif

!########################################################################################################################################
! END STEP 3: FLOW, PHASE FIELD AND PARTICLES INIT
!########################################################################################################################################



!########################################################################################################################################
! START TEMPORAL LOOP: STEP 4 to 9 REPEATED AT EVERY TIME STEP
!########################################################################################################################################

! First step use Euler
alpha=1.0d0
beta=0.0d0
tstart=tstart+1
gamma=1.d0*gumax
!$acc data copyin(piX)
!$acc data create(rhsu_o, rhsv_o, rhsw_o)
!$acc data copyin(mysin, mycos)
#if phiflag == 1
! velocity history for the u^{n+1/2} extrapolation (first step: u^{n-1}=u^n)
!$acc kernels
uold=u
vold=v
wold=w
!$acc end kernels
#endif
call cpu_time(t_start)

! Start temporal loop
do t=tstart,tfin
   ! Create custom label for each marker (profiling)
    write(itcount,'(i4)') t
   ! Range with custom  color (uncomment for profiling)
   ! call nvtxStartRange("Iteration "//itcount,t)
    if (rank.eq.0) write(*,*) "Time step",t,"of",tfin
    call cpu_time(times)
   
   !########################################################################################################################################
   ! START STEP 5: PHASE-FIELD SOLVER (RK4 EXPLICIT) PHASE-FIELD + SURFACTANT (IF ENABLED)
   !########################################################################################################################################
   #if phiflag == 1
   ! 4.2 Get phi at n+1 using RK4 + skew-symmetric splitting
   gamma=1.d0*gumax
   ! Low-storage auxiliary register
   !$acc parallel loop collapse(3)
   do k=1+halo_ext, piX%shape(3)-halo_ext
      do j=1+halo_ext, piX%shape(2)-halo_ext
         do i=1,nx
            q_phi(i,j,k) = 0.d0
            #if surflag == 1
            q_surf(i,j,k) = 0.d0
            #endif
            qfx(i,j,k) = 0.d0
            qfy(i,j,k) = 0.d0
            qfz(i,j,k) = 0.d0
            fbx(i,j,k) = 0.d0
            fby(i,j,k) = 0.d0
            fbz(i,j,k) = 0.d0
         enddo
      enddo
   enddo
   ! store phi^n (with halos, already updated), needed for rho^n in the momentum equation
   !$acc kernels
   phio = phi
   !$acc end kernels

   ! Low storage RK4 - 2 registers - 5 stages - Carpenter-Kennedy 
   do stage = 1, 5
      !$acc host_data use_device(phi)
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, phi, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, phi, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
      !$acc end host_data
      #if surflag == 1
      !$acc host_data use_device(surf)
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, surf, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, surf, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
      !$acc end host_data
      #endif

      ! Compute psidi from current phi, in both halo and non halo cells and then the normals
      !$acc kernels
      do k=1, piX%shape(3)
         do j=1, piX%shape(2)
            do i=1,nx
               val = max(0.d0, min(phi(i,j,k), 1.d0))
               psidi(i,j,k) = eps*log((val+enum)/(1.d0-val+enum))
            enddo
         enddo
      enddo

      ! normal on interio nodes
      do k=1+halo_ext, piX%shape(3)-halo_ext
         do j=1+halo_ext, piX%shape(2)-halo_ext
            do i=1,nx
               ip=i+1 
               im=i-1
               jp=j+1 
               jm=j-1
               kp=k+1 
               km=k-1
               if (ip > nx) ip=1
               if (im < 1)  im=nx
               ! Normals
               normx(i,j,k) = 0.5d0*(psidi(ip,j,k) - psidi(im,j,k))*dxi
               normy(i,j,k) = 0.5d0*(psidi(i,jp,k) - psidi(i,jm,k))*dxi
               normz(i,j,k) = 0.5d0*(psidi(i,j,kp) - psidi(i,j,km))*dxi
               normag = 1.d0/(sqrt(normx(i,j,k)**2 + normy(i,j,k)**2 + normz(i,j,k)**2) + enum)
               normx(i,j,k) = normx(i,j,k)*normag
               normy(i,j,k) = normy(i,j,k)*normag
               normz(i,j,k) = normz(i,j,k)*normag
            enddo
         enddo
      enddo
      !$acc end kernels

      ! Update normales halos, required to then compute normal derivative
      !$acc host_data use_device(normx,normy,normz)
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normx, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normx, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normy, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normy, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normz, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normz, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
      !$acc end host_data

      ! Compute all the fluxes at the faces and add them to rhsphi
      ! skew-symmetric splitting for all contritbuions, all computed as a divergence
      !$acc parallel loop collapse(3) default(present)
      do k=1+halo_ext, piX%shape(3)-halo_ext
         do j=1+halo_ext, piX%shape(2)-halo_ext
            do i=1,nx
               im=i-1
               ip=i+1 
               jm=j-1
               jp=j+1
               km=k-1 
               kp=k+1
               if (ip .gt. nx) ip=1
               if (im .lt. 1)  im=nx
               ! Advection fluxes
               fxp = u(ip,j,k)*0.5d0*(phi(ip,j,k) + phi(i,j,k))
               fxm =  u(i,j,k)*0.5d0*(phi(im,j,k) + phi(i,j,k))
               fyp = v(i,jp,k)*0.5d0*(phi(i,jp,k) + phi(i,j,k))
               fym =  v(i,j,k)*0.5d0*(phi(i,jm,k) + phi(i,j,k))
               fzp = w(i,j,kp)*0.5d0*(phi(i,j,kp) + phi(i,j,k))
               fzm =  w(i,j,k)*0.5d0*(phi(i,j,km) + phi(i,j,k))
               rhsphi(i,j,k) = - (fxp - fxm)*dxi  - (fyp - fym)*dxi - (fzp - fzm)*dxi
               ! total phi flux at the m-faces (u,v,w locations): advective part
               flx = fxm
               fly = fym
               flz = fzm
               ! Diffusion fluxes
               fxp = gamma*eps*(phi(ip,j,k)-phi(i,j,k))*dxi
               fxm = gamma*eps*(phi(i,j,k)-phi(im,j,k))*dxi
               fyp = gamma*eps*(phi(i,jp,k)-phi(i,j,k))*dxi
               fym = gamma*eps*(phi(i,j,k)-phi(i,jm,k))*dxi
               fzp = gamma*eps*(phi(i,j,kp)-phi(i,j,k))*dxi
               fzm = gamma*eps*(phi(i,j,k)-phi(i,j,km))*dxi
               rhsphi(i,j,k) = rhsphi(i,j,k) + (fxp - fxm)*dxi  + (fyp - fym)*dxi + (fzp - fzm)*dxi
               flx = flx - fxm
               fly = fly - fym
               flz = flz - fzm
               ! Sharpening fluxes
               fxp = 0.25d0*gamma*(1.d0-(dtanh(0.25d0*(psidi(ip,j,k)+psidi(i,j,k))*epsi))**2)*0.5d0*(normx(ip,j,k)+normx(i,j,k))
               fxm = 0.25d0*gamma*(1.d0-(dtanh(0.25d0*(psidi(im,j,k)+psidi(i,j,k))*epsi))**2)*0.5d0*(normx(im,j,k)+normx(i,j,k))
               fyp = 0.25d0*gamma*(1.d0-(dtanh(0.25d0*(psidi(i,jp,k)+psidi(i,j,k))*epsi))**2)*0.5d0*(normy(i,jp,k)+normy(i,j,k))
               fym = 0.25d0*gamma*(1.d0-(dtanh(0.25d0*(psidi(i,jm,k)+psidi(i,j,k))*epsi))**2)*0.5d0*(normy(i,jm,k)+normy(i,j,k))
               fzp = 0.25d0*gamma*(1.d0-(dtanh(0.25d0*(psidi(i,j,kp)+psidi(i,j,k))*epsi))**2)*0.5d0*(normz(i,j,kp)+normz(i,j,k))
               fzm = 0.25d0*gamma*(1.d0-(dtanh(0.25d0*(psidi(i,j,km)+psidi(i,j,k))*epsi))**2)*0.5d0*(normz(i,j,km)+normz(i,j,k))
               rhsphi(i,j,k) = rhsphi(i,j,k) - (fxp - fxm)*dxi  - (fyp - fym)*dxi - (fzp - fzm)*dxi
               flx = flx + fxm
               fly = fly + fym
               flz = flz + fzm
               ! accumulate the fluxes with the same RK4 recursion of phi, so that phi^{n+1} = phi^n - dt*div(fb) exactly
               qfx(i,j,k) = rk4a(stage)*qfx(i,j,k) + flx
               qfy(i,j,k) = rk4a(stage)*qfy(i,j,k) + fly
               qfz(i,j,k) = rk4a(stage)*qfz(i,j,k) + flz
               fbx(i,j,k) = fbx(i,j,k) + rk4b(stage)*qfx(i,j,k)
               fby(i,j,k) = fby(i,j,k) + rk4b(stage)*qfy(i,j,k)
               fbz(i,j,k) = fbz(i,j,k) + rk4b(stage)*qfz(i,j,k)
            enddo
         enddo
      enddo

      ! Same approach for surfactant if enabled 
      #if surflag == 1
      !$acc parallel loop collapse(3) default(present)
      do k=1+halo_ext, piX%shape(3)-halo_ext
         do j=1+halo_ext, piX%shape(2)-halo_ext
            do i=1,nx
               im=i-1
               ip=i+1 
               jm=j-1
               jp=j+1
               km=k-1 
               kp=k+1
               if (ip .gt. nx) ip=1
               if (im .lt. 1)  im=nx
               ! Advection fluxes
               fxp = u(ip,j,k)*0.5d0*(surf(ip,j,k) + surf(i,j,k))
               fxm =  u(i,j,k)*0.5d0*(surf(im,j,k) + surf(i,j,k))
               fyp = v(i,jp,k)*0.5d0*(surf(i,jp,k) + surf(i,j,k))
               fym =  v(i,j,k)*0.5d0*(surf(i,jm,k) + surf(i,j,k))
               fzp = w(i,j,kp)*0.5d0*(surf(i,j,kp) + surf(i,j,k))
               fzm =  w(i,j,k)*0.5d0*(surf(i,j,km) + surf(i,j,k))
               rhsurf(i,j,k) = - (fxp - fxm)*dxi  - (fyp - fym)*dxi - (fzp - fzm)*dxi
               ! Diffusion fluxes
               fxp = diffs*(surf(ip,j,k)-surf(i,j,k))*dxi
               fxm = diffs*(surf(i,j,k)-surf(im,j,k))*dxi
               fyp = diffs*(surf(i,jp,k)-surf(i,j,k))*dxi
               fym = diffs*(surf(i,j,k)-surf(i,jm,k))*dxi
               fzp = diffs*(surf(i,j,kp)-surf(i,j,k))*dxi
               fzm = diffs*(surf(i,j,k)-surf(i,j,km))*dxi
               rhsurf(i,j,k) = rhsurf(i,j,k) + (fxp - fxm)*dxi  + (fyp - fym)*dxi + (fzp - fzm)*dxi
               ! Sharpening fluxes
               fxp = -2.d0*diffs*(0.5d0 - 0.5d0*(phi(ip,j,k)+phi(i,j,k)))*0.5d0*(normx(ip,j,k)+normx(i,j,k))*0.5d0*(surf(ip,j,k)+surf(i,j,k))*epsi
               fxm = -2.d0*diffs*(0.5d0 - 0.5d0*(phi(im,j,k)+phi(i,j,k)))*0.5d0*(normx(im,j,k)+normx(i,j,k))*0.5d0*(surf(im,j,k)+surf(i,j,k))*epsi
               fyp = -2.d0*diffs*(0.5d0 - 0.5d0*(phi(i,jp,k)+phi(i,j,k)))*0.5d0*(normy(i,jp,k)+normy(i,j,k))*0.5d0*(surf(i,jp,k)+surf(i,j,k))*epsi
               fym = -2.d0*diffs*(0.5d0 - 0.5d0*(phi(i,jm,k)+phi(i,j,k)))*0.5d0*(normy(i,jm,k)+normy(i,j,k))*0.5d0*(surf(i,jm,k)+surf(i,j,k))*epsi
               fzp = -2.d0*diffs*(0.5d0 - 0.5d0*(phi(i,j,kp)+phi(i,j,k)))*0.5d0*(normz(i,j,kp)+normz(i,j,k))*0.5d0*(surf(i,j,kp)+surf(i,j,k))*epsi
               fzm = -2.d0*diffs*(0.5d0 - 0.5d0*(phi(i,j,km)+phi(i,j,k)))*0.5d0*(normz(i,j,km)+normz(i,j,k))*0.5d0*(surf(i,j,km)+surf(i,j,k))*epsi
               rhsurf(i,j,k) = rhsurf(i,j,k) + (fxp - fxm)*dxi  + (fyp - fym)*dxi + (fzp - fzm)*dxi
            enddo
         enddo
      enddo
      #endif

      ! RK4 update, done after all rhs are computed (in-place update while reading neighbours would be a race)
      !$acc parallel loop collapse(3) default(present)
      do k=1+halo_ext, piX%shape(3)-halo_ext
         do j=1+halo_ext, piX%shape(2)-halo_ext
            do i=1,nx
               q_phi(i,j,k) = rk4a(stage)*q_phi(i,j,k) + dt*rhsphi(i,j,k)
               phi(i,j,k)   = phi(i,j,k) + rk4b(stage)*q_phi(i,j,k)
               #if surflag == 1
               q_surf(i,j,k) = rk4a(stage)*q_surf(i,j,k) + dt*rhsurf(i,j,k)
               surf(i,j,k)   = surf(i,j,k) + rk4b(stage)*q_surf(i,j,k)
               #endif
            enddo
         enddo
      enddo
   enddo  ! end RK4 stages

   ! halos of the time-integrated phi fluxes (needed by the momentum fluxes)
   !$acc host_data use_device(fbx,fby,fbz)
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fbx, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fbx, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fby, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fby, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fbz, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fbz, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   !$acc end host_data

   !$acc host_data use_device(phi)
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, phi, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, phi, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   !$acc end host_data
   #if surflag == 1
   !$acc host_data use_device(surf)
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, surf, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, surf, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   !$acc end host_data
   #endif

   !$acc parallel loop collapse(3)
   do k=1, piX%shape(3)
      do j=1, piX%shape(2)
         do i=1,nx
            val = max(0.0d0, min(phi(i,j,k), 1.0d0))
            psidi(i,j,k) = eps*log((val+enum)/(1.d0-val+enum))
            mu(i,j,k)= muc*(1.d0 - val) + mud*val
         enddo
      enddo
   enddo

   ! recompute normals for NS
   !$acc kernels
   do k=1+halo_ext, piX%shape(3)-halo_ext
      do j=1+halo_ext, piX%shape(2)-halo_ext
         do i=1,nx
            ip=i+1
            jp=j+1
            kp=k+1
            im=i-1
            jm=j-1
            km=k-1
            if (ip .gt. nx) ip=1
            if (im .lt. 1) im=nx
            normx(i,j,k) = 0.5d0*(psidi(ip,j,k) - psidi(im,j,k))*dxi
            normy(i,j,k) = 0.5d0*(psidi(i,jp,k) - psidi(i,jm,k))*dxi
            normz(i,j,k) = 0.5d0*(psidi(i,j,kp) - psidi(i,j,km))*dxi
            normag = 1.d0/(sqrt(normx(i,j,k)*normx(i,j,k) + normy(i,j,k)*normy(i,j,k) + normz(i,j,k)*normz(i,j,k)) + enum)
            normx(i,j,k) = normx(i,j,k)*normag
            normy(i,j,k) = normy(i,j,k)*normag
            normz(i,j,k) = normz(i,j,k)*normag
         enddo
      enddo
   enddo
   !$acc end kernels

   ! Update convective halos + normales halos, required to then compute normal derivative
   !$acc host_data use_device(normx,normy,normz)
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normx, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normx, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normy, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normy, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normz, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, normz, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   !$acc end host_data

   ! call nvtxEndRange
    #endif
   !########################################################################################################################################
   ! END STEP 5: PHASE-FIELD SOLVER 
   !########################################################################################################################################




   !########################################################################################################################################
   ! START STEP 6: USTAR COMPUTATION (PROJECTION STEP)
   !########################################################################################################################################
   ! 6.1 compute rhs 
   ! 6.2 obtain ustar and store old rhs in rhs_o
   ! 6.3 Call halo exchnages along Y and Z for u,v,w

   ! 6.1a Convective and diffusiver terms NS
   ! Loop on inner nodes
   !$acc parallel loop tile(16,4,2) 
   do k=1+halo_ext, piX%shape(3)-halo_ext
      do j=1+halo_ext, piX%shape(2)-halo_ext
         do i=1,nx
            im=i-1
            jm=j-1
            km=k-1
            ! Manual periodicity ony along x (x-pencil), along y and z directions use halos
            if (im .lt. 1) im=nx
            mu12=0.25d0*(mu(i,j,k) + mu(im,j,k) + mu(i,jm,k) + mu(im,jm,k))
            mu13=0.25d0*(mu(i,j,k) + mu(im,j,k) + mu(i,j,km) + mu(im,j,km))
            mu23=0.25d0*(mu(i,j,k) + mu(i,jm,k) + mu(i,j,km) + mu(i,jm,km))
            tau12(i,j,k) = mu12*((u(i,j,k)-u(i,jm,k))*dxi + (v(i,j,k) - v(im,j,k))*dxi) !du/dy + dv/dx
            tau13(i,j,k) = mu13*((u(i,j,k)-u(i,j,km))*dxi + (w(i,j,k) - w(im,j,k))*dxi) !du/dz + dw/dx
            tau23(i,j,k) = mu23*((v(i,j,k)-v(i,j,km))*dxi + (w(i,j,k) - w(i,jm,k))*dxi) !dv/dz + dw/dy
         enddo
      enddo
   enddo

   !$acc host_data use_device(tau12,tau13,tau23)
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, tau12, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, tau12, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, tau13, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, tau13, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, tau23, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, tau23, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   !$acc end host_data

   !$acc parallel loop tile(16,4,2) 
   do k=1+halo_ext, piX%shape(3)-halo_ext
      do j=1+halo_ext, piX%shape(2)-halo_ext
         do i=1,nx
            ip=i+1
            jp=j+1
            kp=k+1
            im=i-1
            jm=j-1
            km=k-1
            ! Manual periodicity ony along x (x-pencil), along y and z directions use halos
            if (ip .gt. nx) ip=1  
            if (im .lt. 1) im=nx
            #if phiflag == 0
            ! compute the products (conservative form)
            h11 = (u(ip,j,k)+u(i,j,k))*(u(ip,j,k)+u(i,j,k))     - (u(i,j,k)+u(im,j,k))*(u(i,j,k)+u(im,j,k))
            h12 = (u(i,jp,k)+u(i,j,k))*(v(i,jp,k)+v(im,jp,k))   - (u(i,j,k)+u(i,jm,k))*(v(i,j,k)+v(im,j,k))
            h13 = (u(i,j,kp)+u(i,j,k))*(w(i,j,kp)+w(im,j,kp))   - (u(i,j,k)+u(i,j,km))*(w(i,j,k)+w(im,j,k))
            h21 = (u(ip,j,k)+u(ip,jm,k))*(v(ip,j,k)+v(i,j,k))   - (u(i,j,k)+u(i,jm,k))*(v(i,j,k)+v(im,j,k))
            h22 = (v(i,jp,k)+v(i,j,k))*(v(i,jp,k)+v(i,j,k))     - (v(i,j,k)+v(i,jm,k))*(v(i,j,k)+v(i,jm,k))
            h23 = (w(i,j,kp)+w(i,jm,kp))*(v(i,j,kp)+v(i,j,k))   - (w(i,j,k)+w(i,jm,k))*(v(i,j,k)+v(i,j,km))
            h31 = (w(ip,j,k)+w(i,j,k))*(u(ip,j,k)+u(ip,j,km))   - (w(i,j,k)+w(im,j,k))*(u(i,j,k)+u(i,j,km))
            h32 = (v(i,jp,k)+v(i,jp,km))*(w(i,jp,k)+w(i,j,k))   - (v(i,j,k)+v(i,j,km))*(w(i,j,k)+w(i,jm,k))
            h33 = (w(i,j,kp)+w(i,j,k))*(w(i,j,kp)+w(i,j,k))     - (w(i,j,k)+w(i,j,km))*(w(i,j,k)+w(i,j,km))
            ! compute the derivative
            h11=h11*0.25d0*dxi
            h12=h12*0.25d0*dxi
            h13=h13*0.25d0*dxi
            h21=h21*0.25d0*dxi
            h22=h22*0.25d0*dxi
            h23=h23*0.25d0*dxi
            h31=h31*0.25d0*dxi
            h32=h32*0.25d0*dxi
            h33=h33*0.25d0*dxi
            ! add to the rhs
            rhsu(i,j,k)=-(h11+h12+h13)
            rhsv(i,j,k)=-(h21+h22+h23)
            rhsw(i,j,k)=-(h31+h32+h33)
            #else
            ! multiphase: convection is computed later in momentum form, with the consistent mass flux
            rhsu(i,j,k)=0.d0
            rhsv(i,j,k)=0.d0
            rhsw(i,j,k)=0.d0
            #endif
            ! viscous terms
            #if phiflag == 0
            ! viscous term (Single-phase)
            h11 = muc*(u(ip,j,k)-2.d0*u(i,j,k)+u(im,j,k))*ddxi
            h12 = muc*(u(i,jp,k)-2.d0*u(i,j,k)+u(i,jm,k))*ddxi
            h13 = muc*(u(i,j,kp)-2.d0*u(i,j,k)+u(i,j,km))*ddxi
            h21 = muc*(v(ip,j,k)-2.d0*v(i,j,k)+v(im,j,k))*ddxi
            h22 = muc*(v(i,jp,k)-2.d0*v(i,j,k)+v(i,jm,k))*ddxi
            h23 = muc*(v(i,j,kp)-2.d0*v(i,j,k)+v(i,j,km))*ddxi
            h31 = muc*(w(ip,j,k)-2.d0*w(i,j,k)+w(im,j,k))*ddxi
            h32 = muc*(w(i,jp,k)-2.d0*w(i,j,k)+w(i,jm,k))*ddxi
            h33 = muc*(w(i,j,kp)-2.d0*w(i,j,k)+w(i,j,km))*ddxi
            #endif
            #if phiflag == 1
            ! viscous term (multiphase, variable viscosity)
            h11 = 2.d0*(mu(i,j,k)*(u(ip,j,k)-u(i,j,k)) - mu(im,j,k)*(u(i,j,k)-u(im,j,k)))*ddxi
            h12 = (tau12(i,jp,k) - tau12(i,j,k))*dxi
            h13 = (tau13(i,j,kp) - tau13(i,j,k))*dxi
            h21 = (tau12(ip,j,k) - tau12(i,j,k))*dxi
            h22 = 2.d0*(mu(i,j,k)*(v(i,jp,k)-v(i,j,k)) - mu(i,jm,k)*(v(i,j,k)-v(i,jm,k)))*ddxi
            h23 = (tau23(i,j,kp) - tau23(i,j,k))*dxi
            h31 = (tau13(ip,j,k) - tau13(i,j,k))*dxi
            h32 = (tau23(i,jp,k) - tau23(i,j,k))*dxi
            h33 = 2.d0*(mu(i,j,k)*(w(i,j,kp)-w(i,j,k)) - mu(i,j,km)*(w(i,j,k)-w(i,j,km)))*ddxi
            #endif
            ! inverse density at the velocity faces, rho^{n+1} = rhoc + (rhod-rhoc)*phi (phi not clipped, to keep mass consistency)
            #if phiflag == 1
            rhoxi = 1.d0/(rhoc + drho*0.5d0*(phi(i,j,k) + phi(im,j,k)))
            rhoyi = 1.d0/(rhoc + drho*0.5d0*(phi(i,j,k) + phi(i,jm,k)))
            rhozi = 1.d0/(rhoc + drho*0.5d0*(phi(i,j,k) + phi(i,j,km)))
            #else
            rhoxi = rhoi
            rhoyi = rhoi
            rhozi = rhoi
            #endif
            ! sum up in rhs
            rhsu(i,j,k)=rhsu(i,j,k)+(h11+h12+h13)*rhoxi
            rhsv(i,j,k)=rhsv(i,j,k)+(h21+h22+h23)*rhoyi
            rhsw(i,j,k)=rhsw(i,j,k)+(h31+h32+h33)*rhozi
            ! NS forcing
            kg = piX%lo(3) + k - 1 
            jg = piX%lo(2) + j - 1
            ! ABC forcing
            rhsu(i,j,k)= rhsu(i,j,k) + f3*mysin(kg)+f2*mycos(jg)
            rhsv(i,j,k)= rhsv(i,j,k) + f1*mysin(i) +f3*mycos(kg)
            rhsw(i,j,k)= rhsw(i,j,k) + f2*mysin(jg)+f1*mycos(i)
         enddo
      enddo
   enddo

   ! Surface tension forces
   #if phiflag == 1
      !$acc kernels
      !Obtain surface tension forces evaluated at the center of the cell (where phi is located)
      do k=1+halo_ext, piX%shape(3)-halo_ext
         do j=1+halo_ext, piX%shape(2)-halo_ext
            do i=1,nx
               ip=i+1
               jp=j+1
               kp=k+1
               im=i-1
               jm=j-1
               km=k-1
               if (ip .gt. nx) ip=1
               if (im .lt. 1) im=nx
               ! continuum-surface force implementation
               curv=0.5d0*(normx(ip,j,k)-normx(im,j,k))*dxi + 0.5d0*(normy(i,jp,k)-normy(i,jm,k))*dxi + 0.5d0*(normz(i,j,kp)-normz(i,j,km))*dxi
               !compute capillary forces: sigma*curvature*gradphi
               fxst(i,j,k)= -sigma*curv*0.5d0*(phi(ip,j,k)-phi(im,j,k))*dxi
               fyst(i,j,k)= -sigma*curv*0.5d0*(phi(i,jp,k)-phi(i,jm,k))*dxi
               fzst(i,j,k)= -sigma*curv*0.5d0*(phi(i,j,kp)-phi(i,j,km))*dxi
            enddo
         enddo
      enddo
      !$acc end kernels

      ! Update halo of fxst, fyst and fzst (required then to interpolate at velocity points)
      !$acc host_data use_device(fxst,fyst,fzst)
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fxst, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fxst, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fyst, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fyst, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fzst, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, fzst, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
      !$acc end host_data 
      
      ! Interpolate force at velocity points
      !$acc kernels
      do k=1+halo_ext, piX%shape(3)-halo_ext
         do j=1+halo_ext, piX%shape(2)-halo_ext
            do i=1,nx
               im=i-1
               jm=j-1
               km=k-1
               if (im .lt. 1) im=nx
               ! inverse density at the velocity faces (same as in the pressure correction)
               rhoxi = 1.d0/(rhoc + drho*0.5d0*(phi(i,j,k) + phi(im,j,k)))
               rhoyi = 1.d0/(rhoc + drho*0.5d0*(phi(i,j,k) + phi(i,jm,k)))
               rhozi = 1.d0/(rhoc + drho*0.5d0*(phi(i,j,k) + phi(i,j,km)))
               rhsu(i,j,k)=rhsu(i,j,k) + 0.5d0*(fxst(im,j,k)+fxst(i,j,k))*rhoxi
               rhsv(i,j,k)=rhsv(i,j,k) + 0.5d0*(fyst(i,jm,k)+fyst(i,j,k))*rhoyi
               rhsw(i,j,k)=rhsw(i,j,k) + 0.5d0*(fzst(i,j,km)+fzst(i,j,k))*rhozi
               ! AB2 increment of the non-convective terms (accelerations), stored in rhs; old rhs in rhs_o
               h11 = rhsu(i,j,k)
               h22 = rhsv(i,j,k)
               h33 = rhsw(i,j,k)
               rhsu(i,j,k) = alpha*h11 - beta*rhsu_o(i,j,k)
               rhsv(i,j,k) = alpha*h22 - beta*rhsv_o(i,j,k)
               rhsw(i,j,k) = alpha*h33 - beta*rhsw_o(i,j,k)
               rhsu_o(i,j,k) = h11
               rhsv_o(i,j,k) = h22
               rhsw_o(i,j,k) = h33
            enddo
         enddo
      enddo
      !$acc end kernels

      ! Consistent mass-momentum transport (Rudman 1997; Mirjalili & Mani 2021 for ACDI):
      ! (rho u)^{n+1} = (rho u)^n - dt*div(mf x u^{n+1/2}), mf = rhoc*u^n + (rhod-rhoc)*fb
      ! fb are the phi fluxes integrated over the RK4 stages, so rho^{n+1} = rho^n - dt*div(mf) holds exactly
      ! Momentum control volumes are centred on the faces: their mass fluxes are averages of the two adjacent cells,
      ! so the same mass balance also holds for the face densities 0.5*(rho(i)+rho(im)).
      ! u* = [(rho u)^n - dt*conv]/rho^{n+1} + dt*AB2(viscous + surface tension + forcing); written to rhs (u is read by neighbours)
      !$acc parallel loop collapse(3) default(present)
      do k=1+halo_ext, piX%shape(3)-halo_ext
         do j=1+halo_ext, piX%shape(2)-halo_ext
            do i=1,nx
               ip=i+1
               jp=j+1
               kp=k+1
               im=i-1
               jm=j-1
               km=k-1
               if (ip .gt. nx) ip=1
               if (im .lt. 1) im=nx
               ! x-momentum (u point, between cells im and i)
               ge = 0.25d0*(MFX(i,j,k)+MFX(ip,j,k))*(UE(i,j,k)+UE(ip,j,k))
               gw = 0.25d0*(MFX(im,j,k)+MFX(i,j,k))*(UE(im,j,k)+UE(i,j,k))
               gn = 0.25d0*(MFY(im,jp,k)+MFY(i,jp,k))*(UE(i,j,k)+UE(i,jp,k))
               gs = 0.25d0*(MFY(im,j,k)+MFY(i,j,k))*(UE(i,jm,k)+UE(i,j,k))
               gt = 0.25d0*(MFZ(im,j,kp)+MFZ(i,j,kp))*(UE(i,j,k)+UE(i,j,kp))
               gb = 0.25d0*(MFZ(im,j,k)+MFZ(i,j,k))*(UE(i,j,km)+UE(i,j,k))
               rhon  = rhoc + drho*0.5d0*(phio(im,j,k) + phio(i,j,k))
               rhonp = rhoc + drho*0.5d0*(phi(im,j,k)  + phi(i,j,k))
               rhsu(i,j,k) = (rhon*u(i,j,k) - dt*(ge-gw+gn-gs+gt-gb)*dxi)/rhonp + dt*rhsu(i,j,k)
               ! y-momentum (v point, between cells jm and j)
               ge = 0.25d0*(MFX(ip,jm,k)+MFX(ip,j,k))*(VE(i,j,k)+VE(ip,j,k))
               gw = 0.25d0*(MFX(i,jm,k)+MFX(i,j,k))*(VE(im,j,k)+VE(i,j,k))
               gn = 0.25d0*(MFY(i,j,k)+MFY(i,jp,k))*(VE(i,j,k)+VE(i,jp,k))
               gs = 0.25d0*(MFY(i,jm,k)+MFY(i,j,k))*(VE(i,jm,k)+VE(i,j,k))
               gt = 0.25d0*(MFZ(i,jm,kp)+MFZ(i,j,kp))*(VE(i,j,k)+VE(i,j,kp))
               gb = 0.25d0*(MFZ(i,jm,k)+MFZ(i,j,k))*(VE(i,j,km)+VE(i,j,k))
               rhon  = rhoc + drho*0.5d0*(phio(i,jm,k) + phio(i,j,k))
               rhonp = rhoc + drho*0.5d0*(phi(i,jm,k)  + phi(i,j,k))
               rhsv(i,j,k) = (rhon*v(i,j,k) - dt*(ge-gw+gn-gs+gt-gb)*dxi)/rhonp + dt*rhsv(i,j,k)
               ! z-momentum (w point, between cells km and k)
               ge = 0.25d0*(MFX(ip,j,km)+MFX(ip,j,k))*(WE(i,j,k)+WE(ip,j,k))
               gw = 0.25d0*(MFX(i,j,km)+MFX(i,j,k))*(WE(im,j,k)+WE(i,j,k))
               gn = 0.25d0*(MFY(i,jp,km)+MFY(i,jp,k))*(WE(i,j,k)+WE(i,jp,k))
               gs = 0.25d0*(MFY(i,j,km)+MFY(i,j,k))*(WE(i,jm,k)+WE(i,j,k))
               gt = 0.25d0*(MFZ(i,j,k)+MFZ(i,j,kp))*(WE(i,j,k)+WE(i,j,kp))
               gb = 0.25d0*(MFZ(i,j,km)+MFZ(i,j,k))*(WE(i,j,km)+WE(i,j,k))
               rhon  = rhoc + drho*0.5d0*(phio(i,j,km) + phio(i,j,k))
               rhonp = rhoc + drho*0.5d0*(phi(i,j,km)  + phi(i,j,k))
               rhsw(i,j,k) = (rhon*w(i,j,k) - dt*(ge-gw+gn-gs+gt-gb)*dxi)/rhonp + dt*rhsw(i,j,k)
            enddo
         enddo
      enddo

      ! store u^n (with halos) for the next extrapolation, then u <- u*
      !$acc kernels
      uold = u
      vold = v
      wold = w
      !$acc end kernels
      !$acc parallel loop collapse(3) default(present)
      do k=1+halo_ext, piX%shape(3)-halo_ext
         do j=1+halo_ext, piX%shape(2)-halo_ext
            do i=1,nx
               u(i,j,k) = rhsu(i,j,k)
               v(i,j,k) = rhsv(i,j,k)
               w(i,j,k) = rhsw(i,j,k)
            enddo
         enddo
      enddo
   #else
      ! 6.2 find u, v and w star only in the inner nodes 
      !$acc kernels
      do k=1+halo_ext, piX%shape(3)-halo_ext
         do j=1+halo_ext, piX%shape(2)-halo_ext
            do i=1,nx
               u(i,j,k) = u(i,j,k) + dt*(alpha*rhsu(i,j,k)-beta*rhsu_o(i,j,k))
               v(i,j,k) = v(i,j,k) + dt*(alpha*rhsv(i,j,k)-beta*rhsv_o(i,j,k))
               w(i,j,k) = w(i,j,k) + dt*(alpha*rhsw(i,j,k)-beta*rhsw_o(i,j,k))
               rhsu_o(i,j,k)=rhsu(i,j,k)
               rhsv_o(i,j,k)=rhsv(i,j,k)
               rhsw_o(i,j,k)=rhsw(i,j,k)
            enddo
         enddo
      enddo
      !$acc end kernels
   #endif

   ! store rhs* in rhs*_o 
   ! First step is done with Euler explicit and then move to AB2 
   alpha=1.5d0
   beta= 0.5d0

   ! 6.3 Dodd & Ferrante (JCP 2014) pressure splitting for variable density:
   ! 1/rho grad(p^{n+1}) ~ 1/rho0 grad(p^{n+1}) + (1/rho - 1/rho0) grad(phat), phat = 2p^n - p^{n-1}
   ! The explicit part is added here to u*, so the Poisson equation keeps constant coefficients (FFT solver):
   ! u* <- u* - dt (1/rho - 1/rho0) grad(phat), then lap(p^{n+1}) = rho0/dt div(u*)
   ! p and pold already have updated halos (p from previous step, pold is a full copy of p)
   #if phiflag == 1
   !$acc parallel loop collapse(3) default(present)
   do k=1+halo_ext, piX%shape(3)-halo_ext
      do j=1+halo_ext, piX%shape(2)-halo_ext
         do i=1,nx
            im=i-1
            jm=j-1
            km=k-1
            if (im .lt. 1) im=nx
            rhoxi = 1.d0/(rhoc + drho*0.5d0*(phi(i,j,k) + phi(im,j,k)))
            rhoyi = 1.d0/(rhoc + drho*0.5d0*(phi(i,j,k) + phi(i,jm,k)))
            rhozi = 1.d0/(rhoc + drho*0.5d0*(phi(i,j,k) + phi(i,j,km)))
            phat_c = 2.d0*p(i,j,k)  - pold(i,j,k)
            phat_x = 2.d0*p(im,j,k) - pold(im,j,k)
            phat_y = 2.d0*p(i,jm,k) - pold(i,jm,k)
            phat_z = 2.d0*p(i,j,km) - pold(i,j,km)
            u(i,j,k) = u(i,j,k) - dt*(rhoxi - rho0i)*(phat_c - phat_x)*dxi
            v(i,j,k) = v(i,j,k) - dt*(rhoyi - rho0i)*(phat_c - phat_y)*dxi
            w(i,j,k) = w(i,j,k) - dt*(rhozi - rho0i)*(phat_c - phat_z)*dxi
         enddo
      enddo
   enddo
   ! store p^n (with halos) before computing p^{n+1}
   !$acc kernels
   pold = p
   !$acc end kernels
   #endif

   ! 5.3 update halos (y and z directions), required to then compute the RHS of Poisson equation because of staggered grid
   !$acc host_data use_device(u,v,w)
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, u, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, u, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, v, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, v, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, w, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, w, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   !$acc end host_data 
   ! call nvtxEndRange
   !########################################################################################################################################
   ! END STEP 6: USTAR COMPUTATION 
   !########################################################################################################################################


   
   !########################################################################################################################################
   ! START STEP 7: POISSON SOLVER FOR PRESSURE
   !########################################################################################################################################
   ! 7.1 Compute rhs of Poisson equation div*ustar: divergence at the cell center 
   !$acc kernels
   do k=1+halo_ext, piX%shape(3)-halo_ext
      do j=1+halo_ext, piX%shape(2)-halo_ext
         do i=1,nx
               ip=i+1
               jp=j+1
               kp=k+1
               if (ip > nx) ip=1
               rhsp(i,j,k) =               (rho0*dxi/dt)*(u(ip,j,k)-u(i,j,k))
               rhsp(i,j,k) = rhsp(i,j,k) + (rho0*dxi/dt)*(v(i,jp,k)-v(i,j,k))
               rhsp(i,j,k) = rhsp(i,j,k) + (rho0*dxi/dt)*(w(i,j,kp)-w(i,j,k))
         enddo
      enddo
   enddo
   !$acc end kernels
   !$acc host_data use_device(rhsp)
   status = cufftExecD2Z(planXf, rhsp, psi_d)
   if (status /= CUFFT_SUCCESS) write(*,*) 'X forward error: ', status
   !$acc end host_data

   ! psi(kx,y,z) -> psi(y,z,kx)
   CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, grid_descD2Z, psi_d, psi_d, work_d_d2z, CUDECOMP_DOUBLE_COMPLEX,piX_d2z%halo_extents, [0,0,0]))
   ! psi(y,z,kx) -> psi(ky,z,kx)
   status = cufftExecZ2Z(planY, psi_d, psi_d, CUFFT_FORWARD)
   if (status /= CUFFT_SUCCESS) write(*,*) 'Y forward error: ', status
   ! psi(ky,z,kx) -> psi(z,kx,ky)
   CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, grid_descD2Z, psi_d, psi_d, work_d_d2z, CUDECOMP_DOUBLE_COMPLEX)) 
   ! psi(z,kx,ky) -> psi(kz,kx,ky)
   status = cufftExecZ2Z(planZ, psi_d, psi_d, CUFFT_FORWARD)
   if (status /= CUFFT_SUCCESS) write(*,*) 'Z forward error: ', status
   ! END of FFT3D forward

   ! call nvtxEndRange
   np(piZ_d2z%order(1)) = piZ_d2z%shape(1)
   np(piZ_d2z%order(2)) = piZ_d2z%shape(2)
   np(piZ_d2z%order(3)) = piZ_d2z%shape(3)
   call c_f_pointer(c_devloc(psi_d), phi3d, piZ_d2z%shape)
   ! divide by -K**2, and normalize
   offsets(piZ_d2z%order(1)) = piZ_d2z%lo(1) - 1
   offsets(piZ_d2z%order(2)) = piZ_d2z%lo(2) - 1
   offsets(piZ_d2z%order(3)) = piZ_d2z%lo(3) - 1
   xoff = offsets(1)
   yoff = offsets(2)
   npx = np(1)
   npy = np(2)
   ! call nvtxStartRange("Solution")
   !$acc kernels
   do jl = 1, npy
      jg = yoff + jl
      do il = 1, npx
         ig = xoff + il
         do k = 1, nz
            !k2 = kx_d(ig)**2 + kx_d(jg)**2 + kx_d(k)**2    
            k2 = (2.d0*(cos(kx_d(ig)*dx)-1.d0) + 2.d0*(cos(kx_d(jg)*dx)-1.d0) + 2.d0*(cos(kx_d(k)*dx)-1.d0))*dxi*dxi !opt: precompute cosines?
            phi3d(k,il,jl) = phi3d(k,il,jl)/k2/(int(nx,8)*int(ny,8)*int(nz,8))
         enddo
      enddo
   enddo
   ! specify mean (corrects division by zero wavenumber above)
   if (xoff == 0 .and. yoff == 0) phi3d(1,1,1) = 0.0
   !$acc end kernels

   ! psi(kz,kx,ky) -> psi(z,kx,ky)
   status = cufftExecZ2Z(planZ, psi_d, psi_d, CUFFT_INVERSE)
   if (status /= CUFFT_SUCCESS) write(*,*) 'Z inverse error: ', status
   ! psi(z,kx,ky) -> psi(ky,z,kx)
   CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, grid_descD2Z, psi_d, psi_d, work_d_d2z, CUDECOMP_DOUBLE_COMPLEX))
   ! psi(ky,z,kx) -> psi(y,z,kx)
   status = cufftExecZ2Z(planY, psi_d, psi_d, CUFFT_INVERSE)
   if (status /= CUFFT_SUCCESS) write(*,*) 'Y inverse error: ', status
   ! psi(y,z,kx) -> psi(kx,y,z)
   CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, grid_descD2Z, psi_d, psi_d, work_d_d2z, CUDECOMP_DOUBLE_COMPLEX,[0,0,0], piX_d2z%halo_extents))
   !$acc host_data use_device(p)
   ! psi(kx,y,z) -> psi(x,y,z)
   status = cufftExecZ2D(planXb, psi_d, p)
   if (status /= CUFFT_SUCCESS) write(*,*) 'X inverse error: ', status
   !$acc end host_data

   !$acc host_data use_device(p)
   ! update halo nodes with pressure (needed for the pressure correction step), using device variable no need to use host-data
   ! Update X-pencil halos in Y and Z direction
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, p, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, p, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   !$acc end host_data 
   ! call nvtxEndRange
   !########################################################################################################################################
   ! END STEP 7: POISSON SOLVER FOR PRESSURE
   !########################################################################################################################################



   !########################################################################################################################################
   ! START STEP 8: VELOCITY CORRECTION
   ! ########################################################################################################################################
   ! 8.1 Correct velocity 
   ! 8.2 Remove mean velocity if using ABC forcing
   ! 8.3 Call halo exchnages along Y and Z for u,v,w
   ! Correct velocity, pressure has also the halo
   ! call nvtxStartRange("Correction")

   !$acc kernels 
   umean=0.d0
   vmean=0.d0
   wmean=0.d0
   do k=1+halo_ext, piX%shape(3)-halo_ext
      do j=1+halo_ext, piX%shape(2)-halo_ext
         do i = 1, piX%shape(1) ! equal to nx (no halo on x)
            im=i-1
            jm=j-1
            km=k-1
            if (im < 1) im=nx
            u(i,j,k)=u(i,j,k) - dt*rho0i*(p(i,j,k)-p(im,j,k))*dxi
            v(i,j,k)=v(i,j,k) - dt*rho0i*(p(i,j,k)-p(i,jm,k))*dxi
            w(i,j,k)=w(i,j,k) - dt*rho0i*(p(i,j,k)-p(i,j,km))*dxi
            umean=umean + u(i,j,k)
            vmean=vmean + v(i,j,k)
            wmean=wmean + w(i,j,k)
          enddo
      enddo
   enddo
   !$acc end kernels 

   ! Remove mean velocity (get local mean of the rank)

   ! Divide by total number of points in the pencil
   umean=umean/nx/(piX%shape(2)-2*halo_ext)/(piX%shape(3)-2*halo_ext)
   vmean=vmean/nx/(piX%shape(2)-2*halo_ext)/(piX%shape(3)-2*halo_ext)
   wmean=wmean/nx/(piX%shape(2)-2*halo_ext)/(piX%shape(3)-2*halo_ext)

   ! Find global mean (MPI_SUM and then divide by number of ranks)
   call MPI_Allreduce(umean,gumean,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD, ierr)
   call MPI_Allreduce(vmean,gvmean,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD, ierr)
   call MPI_Allreduce(wmean,gwmean,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD, ierr)

   ! remove mean value
   !$acc kernels 
   u=u-(gumean/ranks)
   v=v-(gvmean/ranks)
   w=w-(gwmean/ranks)
   !$acc end kernels 

   ! 8.3 update halos (y and z directions), required to then compute the RHS of Poisson equation because of staggered grid
   !$acc host_data use_device(u,v,w)
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, u, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, u, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, v, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, v, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, w, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 2))
   CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, w, work_halo_d, CUDECOMP_DOUBLE, piX%halo_extents, halo_periods, 3))
   !$acc end host_data 

   ! find local maximum velocity
   uc=maxval(u)
   vc=maxval(v)
   wc=maxval(w)
   umax=max(wc,max(uc,vc))
   call MPI_Allreduce(umax,gumax,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD, ierr)
   cou=gumax*dt*dxi
   if (rank.eq.0) then
      write(*,*) "CFL (max among tasks)", cou
      if (cou .gt. 7) stop
      if (.not. ieee_is_finite(cou)) then
         print *, "Error: cou is infinite or NaN!"
         stop
      end if
   endif

   call cpu_time(timef)
   if (rank.eq.0) print '(" Time elapsed = ",f8.1," ms")',1000*(timef-times)

   ! (uncomment for profiling)
   ! call nvtxEndRange

   !########################################################################################################################################
   ! END STEP 8: VELOCITY CORRECTION  
   !########################################################################################################################################


   !########################################################################################################################################
   ! START STEP 9: OUTPUT FIELDS 
   ! ########################################################################################################################################
   if (mod(t,dump) .eq. 0) then
      if (rank .eq. 0) write(*,*) "Saving output files"
      ! write velocity and pressure fiels (1-4)
      call writefield(t,1)
      call writefield(t,2)
      call writefield(t,3)
      !call writefield(t,4)
      #if phiflag == 1
      call writefield(t,5)
      #endif
      #if surflag == 1
      call writefield(t,6)
      #endif
   endif
   !########################################################################################################################################
   ! END STEP 9: OUTPUT FIELDS N  
   !########################################################################################################################################

! (uncomment for profiling)
! call nvtxEndRange

enddo
call cpu_time(t_end)
elapsed = t_end-t_start
if (rank .eq. 0) write(*,*)  'Elapsed time (seconds):', elapsed
!$acc end data
!$acc end data
!$acc end data



! Remove allocated variables (add new)
deallocate(x_ext)
deallocate(u,v,w)
!deallocate(tanh_psi, mysin, mycos)
deallocate(rhsu,rhsv,rhsw)
deallocate(rhsu_o,rhsv_o,rhsw_o)
!deallocate(phi,rhsphi,normx,normy,normz)
!deallocate(normx_f, normy_f, normz_f)
!deallocate(fxst, fyst, fzst)
!deallocate(phi_tmp, rhsphik2, rhsphik3, rhsphik4)
!deallocate(psidi, tanh_psi)

call mpi_finalize(ierr)

end program main

