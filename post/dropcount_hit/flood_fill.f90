! Iterative 26-connected flood fill using an explicit DFS stack.
! Flat index encoding: flat = (k-1)*nx*nx + (j-1)*nx + (i-1), range [0, nx^3-1].
! stack workspace (size >= nx*nx*nx) is allocated once by the caller and reused.
! Computes centroid (cx,cy,cz) and the inertia tensor centred on the centroid
! (Ixx,Iyy,Izz,Ixy,Ixz,Iyz) in physical units (dx^5, assuming unit density).
! Periodic wrapping is handled via minimum-image unwrapping relative to the seed
! voxel, so droplets split at the boundary give the correct centroid and tensor.
! Assumption: no droplet spans more than half the domain in any direction.
subroutine flood_fill_iter(top, label, stack, current_label, i0, j0, k0, &
                           cluster_size, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
  use param
  implicit none
  integer, intent(in)    :: top(nx,nx,nx)
  integer, intent(inout) :: label(nx,nx,nx)
  integer, intent(inout) :: stack(*)
  integer, intent(in)    :: current_label, i0, j0, k0
  integer, intent(out)   :: cluster_size
  double precision, intent(out) :: cx, cy, cz
  double precision, intent(out) :: Ixx, Iyy, Izz, Ixy, Ixz, Iyz

  integer :: sp, flat, i, j, k, ni, nj, nk, d
  double precision :: x0, y0, z0, xi, yj, zk, N
  double precision :: sum_x, sum_y, sum_z
  double precision :: sum_x2, sum_y2, sum_z2, sum_xy, sum_xz, sum_yz

  ! 26-connectivity offsets: 6 faces + 12 edges + 8 corners
  integer, parameter :: ndirs = 26
  integer :: ox(ndirs), oy(ndirs), oz(ndirs)

  data ox / 1,-1, 0, 0, 0, 0,                    &  ! faces
            1, 1,-1,-1, 1, 1,-1,-1, 0, 0, 0, 0,  &  ! edges
            1, 1, 1, 1,-1,-1,-1,-1 /                 ! corners
  data oy / 0, 0, 1,-1, 0, 0,                    &
            1,-1, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1,  &
            1, 1,-1,-1, 1, 1,-1,-1 /
  data oz / 0, 0, 0, 0, 1,-1,                    &
            0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1,  &
            1,-1, 1,-1, 1,-1, 1,-1 /

  ! Physical seed coordinates (reference for minimum-image unwrapping)
  x0 = dble(i0-1) * dx
  y0 = dble(j0-1) * dx
  z0 = dble(k0-1) * dx

  sum_x  = 0.0d0;  sum_y  = 0.0d0;  sum_z  = 0.0d0
  sum_x2 = 0.0d0;  sum_y2 = 0.0d0;  sum_z2 = 0.0d0
  sum_xy = 0.0d0;  sum_xz = 0.0d0;  sum_yz = 0.0d0

  label(i0,j0,k0) = current_label
  sp           = 1
  stack(1)     = (k0-1)*nx*nx + (j0-1)*nx + (i0-1)
  cluster_size = 0

  do while (sp > 0)
    flat = stack(sp)
    sp   = sp - 1
    cluster_size = cluster_size + 1

    ! decode flat (0-based) to 1-based indices
    i = mod(flat,       nx) + 1
    j = mod(flat/nx,    nx) + 1
    k =     flat/(nx*nx)   + 1

    ! Physical coordinates, unwrapped relative to seed via minimum image
    xi = dble(i-1) * dx;  xi = xi - lx * nint((xi - x0) / lx)
    yj = dble(j-1) * dx;  yj = yj - lx * nint((yj - y0) / lx)
    zk = dble(k-1) * dx;  zk = zk - lx * nint((zk - z0) / lx)

    sum_x  = sum_x  + xi
    sum_y  = sum_y  + yj
    sum_z  = sum_z  + zk
    sum_x2 = sum_x2 + xi*xi
    sum_y2 = sum_y2 + yj*yj
    sum_z2 = sum_z2 + zk*zk
    sum_xy = sum_xy + xi*yj
    sum_xz = sum_xz + xi*zk
    sum_yz = sum_yz + yj*zk

    do d = 1, ndirs
      ni = mod(i - 1 + ox(d) + nx, nx) + 1
      nj = mod(j - 1 + oy(d) + nx, nx) + 1
      nk = mod(k - 1 + oz(d) + nx, nx) + 1
      if (top(ni,nj,nk) == 1 .and. label(ni,nj,nk) == 0) then
        label(ni,nj,nk) = current_label
        sp       = sp + 1
        stack(sp) = (nk-1)*nx*nx + (nj-1)*nx + (ni-1)
      end if
    end do
  end do

  N  = dble(cluster_size)
  cx = sum_x / N
  cy = sum_y / N
  cz = sum_z / N

  ! Inertia tensor centred on the centroid via the parallel-axis theorem.
  ! All coordinates are already in the unwrapped frame, so no wrapping artefacts.
  ! Each voxel carries mass dx^3 (unit density); units are dx^5.
  Ixx = dx**3 * ((sum_y2 + sum_z2) - N*(cy**2 + cz**2))
  Iyy = dx**3 * ((sum_x2 + sum_z2) - N*(cx**2 + cz**2))
  Izz = dx**3 * ((sum_x2 + sum_y2) - N*(cx**2 + cy**2))
  Ixy = -dx**3 * (sum_xy - N*cx*cy)
  Ixz = -dx**3 * (sum_xz - N*cx*cz)
  Iyz = -dx**3 * (sum_yz - N*cy*cz)

  ! Wrap centroid back to [0, lx)
  cx = cx - lx * floor(cx / lx)
  cy = cy - lx * floor(cy / lx)
  cz = cz - lx * floor(cz / lx)

end subroutine flood_fill_iter
