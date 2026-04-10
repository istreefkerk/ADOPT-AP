MODULE uz_sz_interaction
IMPLICIT NONE
CONTAINS
SUBROUTINE call_update_soil(z, zroot_u, zroot_l, theta_sat_u, theta_fc_u,&
theta_u, theta_sat_l, theta_fc_l, theta_l, deltaS, Sy, head)
!INTEGER, INTENT(IN) :: z(:)
REAL, INTENT(IN) :: z(:)
REAL, INTENT(IN) :: zroot_u(:)
REAL, INTENT(IN) :: zroot_l(:)
REAL, INTENT(IN) :: theta_sat_u(:)
REAL, INTENT(IN) :: theta_fc_u(:)
REAL, INTENT(IN) :: theta_u(:)
REAL, INTENT(IN) :: theta_sat_l(:)
REAL, INTENT(IN) :: theta_fc_l(:)
REAL, INTENT(IN) :: theta_l(:)
REAL, INTENT(IN) :: deltaS(:)
REAL, INTENT(IN) :: Sy(:)

REAL, INTENT(INOUT) :: head(:)

INTEGER :: i, npoint, npointtp
REAL :: ihead

npoint = SIZE(s)
npointtp = SIZE(theta_sat_u)

! Other variables
!INTEGER :: i, donor, recvr, npoint
!REAL :: Q_out, Q_TLp, Q_inip, Q_aofp

! Check if saturated water content of the upper layer is provides
IF npointtp .eq. 1 THEN

DO i = 1, npoint, 1

CALL UPDATE_2LAYER_SOIL(z(i), zroot_u(i), zroot_l(i), theta_sat_u(i),&
theta_fc_u(i), theta_u(i), theta_sat_l(i), theta_fc_l(i), theta_l(i),&
deltaS(i), Sy(i), Sy(i), head(i))

END DO

ELSE
! Check if saturated water content of the upper layer is provides
DO i = 1, npoint, 1

CALL UPDATE_2LAYER_SOIL(z(i), zroot_u(i), zroot_l(i), theta_sat_u,&
theta_fc_u, theta_u, theta_sat_l(i), theta_fc_l(i), theta_l(i),&
deltaS(i), Sy(i), ihead)
     
END DO

END SUBROUTINE

!======================================================================
! SUBROUTINE for layered soil parameters
!======================================================================	

SUBROUTINE UPDATE_2LAYER_SOIL(z, zroot_u, zroot_l, theta_sat_u,&
theta_fc_u, theta_u, theta_sat_l, theta_fc_l, theta_l, deltaS, Sy,&
head)
   
!""" Function to calculate soil gorundwater interaction,
!this function update soil storage when the water table
!raise or decrease.
!INPUTS:
!-------
!z:	surface elevation
!zroot_u:	root elevation top layer
!zroot_l:	root elevation bottom layer
!theta_sat_u:	
!theta_fc_u:
!theta_u:
!theta_sat_l:
!theta_fc_l:
!theta_l:
!delta:	change in storage
!head:	water table elevation
!Sy:		aquifer specific yield
!OUTPUTS:
!--------
!h: water table elevation
!"""

REAL, INTENT(IN) :: z, zroot_u, zroot_l, theta_sat_u, theta_fc_u, theta_u
REAL, INTENT(IN) :: theta_sat_l, theta_fc_l, theta_l, deltaS, Sy
REAL, INTENT(INOUT) :: head

INTEGER :: i,
REAL :: t,

IF deltaS .lt. THEN
!when water table decrease
IF head .gt. zroot_u THEN
deltaSu = (head-zroot_u)*(theta_sat_u-theta_fc_u)
deltaSl = (zroot_u-zroot_l)*(theta_sat_l-theta_fc_l)
ELSE
deltaSu = 0
deltaSl = (head-zroot_l)*(theta_sat_l-theta_fc_l)
IF head .lt. zroot_l THEN
deltaSl = 0
END IF
END IF

deltaS = abs(deltaS)		
deltaSp = deltaS - deltaSu
!print(deltaSp, deltaSl, deltaSl)
! update head elevation
IF deltaSu .gt. 0 THEN
! initial water table located in the upper soil layer
IF deltaSp .lt. 0 THEN
! water table always within the upper soil layer
head = head - deltaS/(theta_sat_u-theta_fc_u)
ELSE
deltaSpp = deltaS - deltaSu - deltaSl
IF deltaSpp .lt. 0 THEN
! water table falls to lower soil layer
head = zroot_u - (deltaS-deltaSu)/(theta_sat_u-theta_fc_u)
ELSE
! water table fails below the lower soil layer
head = zroot_l - deltaSpp/(Sy)
END IF
END IF
ELSE
! water table located below the upper soil layer
deltaSp = deltaS - deltaSl
IF deltaSl .gt. 0 THEN
! water table always located below the upper soil layer
IF deltaSp .lt. 0 THEN
! water table always within the lower soil layer
head = head - (deltaS)/(theta_sat_l-theta_fc_l)
ELSE
!water table falls below the lower layer
head = zroot_u - (deltaS-deltaSl)/Sy
END IF
ELSE
! water table allways in the aquifer
head = head - deltaS/Sy
END IF
!print(head)	
END IF
ELSE
! when water table increases
IF head .gt. zroot_u THEN
deltaSl = 0
deltaSa = 0
ELSE
IF head .gt. zroot_l THEN
deltaSa = 0
deltaSl = (zroot_u-head)*(theta_sat_l-theta_l)
ELSE
deltaSl = (zroot_u-zroot_l)*(theta_sat_l-theta_l)
deltaSa = (zroot_l-head)*Sy
END IF
END IF
! update water table
deltaSp = deltaS - deltaSa

IF deltaSa .gt. 0 THEN
! initial water table in the aquifer
IF deltaSp .lt. 0 THEN
! water table always below the lower soil layer
head = head + deltaS/Sy
ELSE
! water table always above the aquifer
deltaSpp = deltaSp - deltaSl
IF deltaSpp .gt. 0 THEN
! water table rises above the lower layer
head = zroot_u + (deltaS-deltaSa)/(theta_sat_l-theta_l)
ELSE
! water table rises above the aquifer
head = zroot_l + deltaSpp/(theta_sat_u-theta_u)
END IF
END IF
ELSE
! initial water table above the aquifer
IF deltaSl .gt. 0 THEN
deltaSpp = deltaS - deltaSl
IF deltaSpp .gt. 0 THEN
! water table within the lower layer
head = head + deltaS/(theta_sat_l-theta_l)
ELSE
! water table rises to upper layer
head = zroot_u + deltaSpp/(theta_sat_u-theta_u)
END IF
ELSE
! water table always in the upper layer
head = head + deltaS/(theta_sat_u-theta_u)
END IF
END IF
END IF			
END SUBROUTINE