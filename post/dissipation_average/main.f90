! ======================================================================
! main.f90
! ----------------------------------------------------------------------
! Computes the volume-averaged viscous dissipation
!     <eps> = < 2 * nu(phi) * s_ij s_ij >,   s_ij = 1/2 (du_i/dx_j + du_j/dx_i)
! on a staggered (MAC) grid in triply-periodic HIT.
!
! Staggered layout assumed:
!     u(i,j,k)   at face (i-1/2, j,   k  )
!     v(i,j,k)   at face (i,   j-1/2, k  )
!     w(i,j,k)   at face (i,   j,   k-1/2)
!     phi(i,j,k) at cell center (i, j, k)
!
! Output: prints <eps> to stdout for each snapshot.
!
! Build (gfortran):
!     gfortran -O3 -march=native -fopenmp main.f90 -o compute_diss
! ======================================================================

module commondata
  integer :: nx
  integer :: nstart, nend, ndump
  double precision, parameter :: pi = 3.14159265358979d0
  double precision :: dx, lx, dxi
  double precision, allocatable, dimension(:,:,:) :: u, v, w
  double precision, allocatable, dimension(:,:,:) :: dudx, dudy, dudz
  double precision, allocatable, dimension(:,:,:) :: dvdx, dvdy, dvdz
  double precision, allocatable, dimension(:,:,:) :: dwdx, dwdy, dwdz

  character(len=*), parameter :: input_dir = '../../hit/output'
  ! Phase convention: .true. if phi=1 inside drops, .false. if phi=0 inside.
  logical, parameter :: phi_drop_is_one = .true.
  ! Two-viscosity setup. For matched case set nu_drop = nu_matrix.
  double precision, parameter :: nu_drop   = 0.006d0
  double precision, parameter :: nu_matrix = 0.006d0
end module commondata


program compute_diss_staggered
  use commondata
  implicit none
  integer :: nstep

  nx    = 512
  lx    = 2.0d0 * pi
  dx    = lx / dble(nx)
  dxi   = 1.0d0 / dx

  nstart = 200000
  ndump  = 5000
  nend   = 200000

  write(*,'(a,i0,a,i0,a,i0)') '[info] grid: ', nx, ' x ', nx, ' x ', nx
  write(*,'(a,es12.5)')        '[info] dx   = ', dx
  write(*,'(a,es12.5,a,es12.5)') '[info] nu_drop = ', nu_drop, &
       '   nu_matrix = ', nu_matrix
  write(*,'(a,a)')             '[info] input_dir = ', trim(input_dir)
  write(*,'(a,i0,a,i0,a,i0)') '[info] snapshots: nstart=', nstart, &
       ' ndump=', ndump, ' nend=', nend

  allocate(u(nx,nx,nx), v(nx,nx,nx), w(nx,nx,nx))
  allocate(dudx(nx,nx,nx), dudy(nx,nx,nx), dudz(nx,nx,nx))
  allocate(dvdx(nx,nx,nx), dvdy(nx,nx,nx), dvdz(nx,nx,nx))
  allocate(dwdx(nx,nx,nx), dwdy(nx,nx,nx), dwdz(nx,nx,nx))

  do nstep = nstart, nend, ndump
     call process_snapshot(nstep)
  end do

  deallocate(u, v, w)
  deallocate(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
  write(*,'(a)') '[done]'

end program compute_diss_staggered


! ----------------------------------------------------------------------
subroutine process_snapshot(nstep)
  use commondata
  implicit none
  integer, intent(in) :: nstep

  double precision, allocatable :: phi(:,:,:)
  double precision :: eps_mean, sum_e, max_div, div_loc
  integer(kind=8) :: n_tot
  integer :: i, j, k, ip, jp, kp, im, jm, km
  double precision :: nu_loc, sij_sij_loc, s11, s22, s33, s12, s13, s23
  double precision :: phi_d, eps_loc

  allocate(phi(nx,nx,nx))

  write(*,'(a,i0)') '[step] reading snapshot ', nstep
  call read_fields_snap(nstep, phi)

  ! Diagonal derivatives: forward differences (cell-centered result)
  !$omp parallel do default(none) shared(u,v,w,dudx,dvdy,dwdz,nx,dxi) &
  !$omp private(i,j,k,ip,jp,kp)
  do k = 1, nx
     kp = k + 1; if (kp > nx) kp = 1
     do j = 1, nx
        jp = j + 1; if (jp > nx) jp = 1
        do i = 1, nx
           ip = i + 1; if (ip > nx) ip = 1
           dudx(i,j,k) = (u(ip,j ,k ) - u(i,j,k)) * dxi
           dvdy(i,j,k) = (v(i ,jp,k ) - v(i,j,k)) * dxi
           dwdz(i,j,k) = (w(i ,j ,kp) - w(i,j,k)) * dxi
        end do
     end do
  end do
  !$omp end parallel do

  ! Off-diagonal derivatives at edges, then averaged to cell centers below
  !$omp parallel do default(none) shared(u,v,w,dudy,dudz,dvdx,dvdz,dwdx,dwdy,nx,dxi) &
  !$omp private(i,j,k,im,jm,km)
  do k = 1, nx
     km = k - 1; if (km < 1) km = nx
     do j = 1, nx
        jm = j - 1; if (jm < 1) jm = nx
        do i = 1, nx
           im = i - 1; if (im < 1) im = nx
           dudy(i,j,k) = (u(i ,j ,k ) - u(i ,jm,k )) * dxi
           dudz(i,j,k) = (u(i ,j ,k ) - u(i ,j ,km)) * dxi
           dvdx(i,j,k) = (v(i ,j ,k ) - v(im,j ,k )) * dxi
           dvdz(i,j,k) = (v(i ,j ,k ) - v(i ,j ,km)) * dxi
           dwdx(i,j,k) = (w(i ,j ,k ) - w(im,j ,k )) * dxi
           dwdy(i,j,k) = (w(i ,j ,k ) - w(i ,jm,k )) * dxi
        end do
     end do
  end do
  !$omp end parallel do

  call average_edges_to_center(dudy, 1, 2)
  call average_edges_to_center(dudz, 1, 3)
  call average_edges_to_center(dvdx, 2, 1)
  call average_edges_to_center(dvdz, 2, 3)
  call average_edges_to_center(dwdx, 3, 1)
  call average_edges_to_center(dwdy, 3, 2)

  max_div = 0.0d0
  sum_e   = 0.0d0
  n_tot   = 0

  !$omp parallel do default(none) &
  !$omp shared(dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz,phi,nx) &
  !$omp private(i,j,k,s11,s22,s33,s12,s13,s23,sij_sij_loc,nu_loc,phi_d,eps_loc,div_loc) &
  !$omp reduction(max:max_div) &
  !$omp reduction(+:sum_e,n_tot)
  do k = 1, nx
     do j = 1, nx
        do i = 1, nx
           s11 = dudx(i,j,k)
           s22 = dvdy(i,j,k)
           s33 = dwdz(i,j,k)
           s12 = 0.5d0 * (dudy(i,j,k) + dvdx(i,j,k))
           s13 = 0.5d0 * (dudz(i,j,k) + dwdx(i,j,k))
           s23 = 0.5d0 * (dvdz(i,j,k) + dwdy(i,j,k))
           sij_sij_loc = s11*s11 + s22*s22 + s33*s33 &
                       + 2.0d0 * (s12*s12 + s13*s13 + s23*s23)

           if (phi_drop_is_one) then
              phi_d = phi(i,j,k)
           else
              phi_d = 1.0d0 - phi(i,j,k)
           end if
           if (phi_d < 0.0d0) phi_d = 0.0d0
           if (phi_d > 1.0d0) phi_d = 1.0d0
           nu_loc = nu_drop * phi_d + nu_matrix * (1.0d0 - phi_d)

           eps_loc = 2.0d0 * nu_loc * sij_sij_loc
           sum_e   = sum_e + eps_loc
           n_tot   = n_tot + 1

           div_loc = abs(s11 + s22 + s33)
           if (div_loc > max_div) max_div = div_loc
        end do
     end do
  end do
  !$omp end parallel do

  eps_mean = sum_e / dble(n_tot)
  write(*,'(a,i0,a,es16.8,a,es10.3)') &
       '[step ', nstep, '] <eps> = ', eps_mean, '   max|div u| = ', max_div

  deallocate(phi)
end subroutine process_snapshot


! ----------------------------------------------------------------------
subroutine read_fields_snap(nstep, phi)
  use commondata
  implicit none
  integer, intent(in) :: nstep
  double precision, intent(out) :: phi(nx,nx,nx)
  character(len=512) :: fname
  integer :: io, ierr

  write(fname,'(a,a,a,i8.8,a)') trim(input_dir), '/', 'u_', nstep, '.dat'
  open(newunit=io, file=trim(fname), form='unformatted', &
       access='stream', status='old', action='read', iostat=ierr)
  if (ierr /= 0) then; write(*,*) '[error] cannot open ', trim(fname); stop 1; end if
  read(io) u; close(io)

  write(fname,'(a,a,a,i8.8,a)') trim(input_dir), '/', 'v_', nstep, '.dat'
  open(newunit=io, file=trim(fname), form='unformatted', &
       access='stream', status='old', action='read', iostat=ierr)
  if (ierr /= 0) then; write(*,*) '[error] cannot open ', trim(fname); stop 1; end if
  read(io) v; close(io)

  write(fname,'(a,a,a,i8.8,a)') trim(input_dir), '/', 'w_', nstep, '.dat'
  open(newunit=io, file=trim(fname), form='unformatted', &
       access='stream', status='old', action='read', iostat=ierr)
  if (ierr /= 0) then; write(*,*) '[error] cannot open ', trim(fname); stop 1; end if
  read(io) w; close(io)

  write(fname,'(a,a,a,i8.8,a)') trim(input_dir), '/', 'phi_', nstep, '.dat'
  open(newunit=io, file=trim(fname), form='unformatted', &
       access='stream', status='old', action='read', iostat=ierr)
  if (ierr /= 0) then
     ! No phase-field dump for this snapshot (e.g. single-phase HIT precursor run):
     ! fall back to phi = 0 everywhere, i.e. pure matrix phase (nu = nu_matrix).
     write(*,'(a,a,a)') '[warn] phi field not found (', trim(fname), &
          '); assuming single-phase run, using nu_matrix everywhere'
     phi = 0.0d0
  else
     read(io) phi; close(io)
  end if
end subroutine read_fields_snap


! ----------------------------------------------------------------------
! Average an array of edge-centered derivatives to cell centers, in place.
!
! face_axis  : axis along which the source velocity is face-staggered (1, 2, or 3)
! deriv_axis : axis along which the derivative was taken
! ----------------------------------------------------------------------
subroutine average_edges_to_center(arr, face_axis, deriv_axis)
  use commondata
  implicit none
  double precision, intent(inout) :: arr(nx,nx,nx)
  integer, intent(in) :: face_axis, deriv_axis

  double precision, allocatable :: tmp(:,:,:)
  integer :: i, j, k
  integer :: i_fp, j_fp, k_fp
  integer :: i_dp, j_dp, k_dp
  integer :: i_bp, j_bp, k_bp

  allocate(tmp(nx,nx,nx))

  !$omp parallel do default(none) shared(arr,tmp,nx,face_axis,deriv_axis) &
  !$omp private(i,j,k,i_fp,j_fp,k_fp,i_dp,j_dp,k_dp,i_bp,j_bp,k_bp)
  do k = 1, nx
     do j = 1, nx
        do i = 1, nx
           i_fp = i; j_fp = j; k_fp = k
           i_dp = i; j_dp = j; k_dp = k
           select case (face_axis)
              case (1); i_fp = i + 1; if (i_fp > nx) i_fp = 1
              case (2); j_fp = j + 1; if (j_fp > nx) j_fp = 1
              case (3); k_fp = k + 1; if (k_fp > nx) k_fp = 1
           end select
           select case (deriv_axis)
              case (1); i_dp = i + 1; if (i_dp > nx) i_dp = 1
              case (2); j_dp = j + 1; if (j_dp > nx) j_dp = 1
              case (3); k_dp = k + 1; if (k_dp > nx) k_dp = 1
           end select
           i_bp = i_fp; j_bp = j_fp; k_bp = k_fp
           select case (deriv_axis)
              case (1); i_bp = i_bp + 1; if (i_bp > nx) i_bp = 1
              case (2); j_bp = j_bp + 1; if (j_bp > nx) j_bp = 1
              case (3); k_bp = k_bp + 1; if (k_bp > nx) k_bp = 1
           end select
           tmp(i,j,k) = 0.25d0 * ( arr(i,   j,   k  ) &
                                 + arr(i_fp,j_fp,k_fp) &
                                 + arr(i_dp,j_dp,k_dp) &
                                 + arr(i_bp,j_bp,k_bp) )
        end do
     end do
  end do
  !$omp end parallel do

  arr = tmp
  deallocate(tmp)
end subroutine average_edges_to_center
