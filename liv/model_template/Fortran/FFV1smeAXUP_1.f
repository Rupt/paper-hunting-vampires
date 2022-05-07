C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Gamma(3,2,1)
C     
      SUBROUTINE FFV1smeAXUP_1(F2, V3, COUP, M1, W1,F1)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 F1(6)
      COMPLEX*16 F2(*)
      REAL*8 M1
      REAL*8 P1(0:3)
      COMPLEX*16 V3(*)
      REAL*8 W1
      COMPLEX*16 DENOM

      COMPLEX*16 tmp12
      COMPLEX*16 tmp13
      COMPLEX*16 tmp14
      COMPLEX*16 tmp15
      COMPLEX*16 tmp16
      COMPLEX*16 tmp17
      COMPLEX*16 tmp18
      COMPLEX*16 tmp19

      COMPLEX*16 ca00
      COMPLEX*16 ca01
      COMPLEX*16 ca02
      COMPLEX*16 ca03
      COMPLEX*16 ca10
      COMPLEX*16 ca11
      COMPLEX*16 ca12
      COMPLEX*16 ca13
      COMPLEX*16 ca20
      COMPLEX*16 ca21
      COMPLEX*16 ca22
      COMPLEX*16 ca23
      COMPLEX*16 ca30
      COMPLEX*16 ca31
      COMPLEX*16 ca32
      COMPLEX*16 ca33

C   Define the sme coefficients

      ca00 = (0D0,0D0)
      ca01 = (0.05D0,-0.43D0)
      ca02 = (0D0,0D0)
      ca03 = (0D0,0D0)
      ca10 = (0.05D0,0.43D0)
      ca11 = (0D0,0D0)
      ca12 = (-1.88D0,0.72D0)
      ca13 = (0D0,0D0)
      ca20 = (0D0,0.0D0)
      ca21 = (-1.88D0,-0.72D0)
      ca22 = (0D0,0D0)
      ca23 = (0D0,0D0)
      ca30 = (0D0,0.0D0)
      ca31 = (0D0,0.0D0)
      ca32 = (0D0,0.0D0)
      ca33 = (0D0,0D0)

      F1(1) = +F2(1)+V3(1)
      F1(2) = +F2(2)+V3(2)
      P1(0) = -DBLE(F1(1))
      P1(1) = -DBLE(F1(2))
      P1(2) = -DIMAG(F1(2))
      P1(3) = -DIMAG(F1(1))

C   Define some tmp variables

      tmp12=ca00*V3(3)+ca01*V3(4)+ca02*V3(5)+ca03*V3(6)
      tmp13=ca10*V3(3)+ca11*V3(4)+ca12*V3(5)+ca13*V3(6)
      tmp14=ca20*V3(3)+ca21*V3(4)+ca22*V3(5)+ca23*V3(6)
      tmp15=ca30*V3(3)+ca31*V3(4)+ca32*V3(5)+ca33*V3(6)
      tmp16=P1(0)+P1(3)
      tmp17=P1(0)-P1(3)
      tmp18=P1(1)+CI*P1(2)
      tmp19=P1(1)-CI*P1(2)


      DENOM = COUP/(P1(0)**2-P1(1)**2-P1(2)**2-P1(3)**2 - M1 * (M1 -CI
     $ * W1))
        
      F1(3) = DENOM*(0.5D0)*( 
     $ (-(tmp14*tmp18)-(CI)*tmp13*tmp18-(CI)*tmp12*tmp16-(CI)*tmp15*tmp16)*F2(3)
     $ +((CI)*tmp15*tmp18-(CI)*tmp12*tmp18-(CI)*tmp13*tmp16+tmp14*tmp16)*F2(4)
     $ +(CI)*M1*(-tmp12+tmp15)*F2(5)+M1*((CI)*tmp13-tmp14)*F2(6))

      F1(4) = DENOM*(0.5D0)*(
     $ (-(CI)*tmp12*tmp19-(CI)*tmp15*tmp19-(CI)*tmp13*tmp17-tmp14*tmp17)*F2(3)
     $ +(-(CI)*tmp13*tmp19+tmp14*tmp19+(CI)*tmp15*tmp17-(CI)*tmp12*tmp17)*F2(4)
     $ +(tmp14+(CI)*tmp13)*M1*F2(5)-(CI)*(tmp12+tmp15)*M1*F2(6))

      F1(5) = DENOM*(0.5D0)*(
     $ (CI)*(tmp12+tmp15)*M1*F2(3)+M1*((CI)*tmp13-tmp14)*F2(4)
     $ +((CI)*tmp13*tmp18+tmp14*tmp18+(CI)*tmp12*tmp17-(CI)*tmp15*tmp17)*F2(5)
     $ +(-(CI)*tmp12*tmp18-(CI)*tmp15*tmp18+tmp14*tmp17-(CI)*tmp13*tmp17)*F2(6))

      F1(6) = DENOM*(0.5D0)*(
     $ (tmp14+(CI)*tmp13)*M1*F2(3)+(CI)*(tmp12-tmp15)*M1*F2(4)
     $ +(-(CI)*tmp12*tmp19+(CI)*tmp15*tmp19-(CI)*tmp13*tmp16-tmp14*tmp16)*F2(5)
     $ +(-(tmp14*tmp19)+(CI)*tmp13*tmp19+(CI)*tmp12*tmp16+(CI)*tmp15*tmp16)*F2(6))

      END


      SUBROUTINE FFV1smeAXUPP1N_1(F2, V3, COUP,F1)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 F1(6)
      COMPLEX*16 F2(*)
      COMPLEX*16 V3(*)

      COMPLEX*16 tmp12
      COMPLEX*16 tmp13
      COMPLEX*16 tmp14
      COMPLEX*16 tmp15

      COMPLEX*16 ca00
      COMPLEX*16 ca01
      COMPLEX*16 ca02
      COMPLEX*16 ca03
      COMPLEX*16 ca10
      COMPLEX*16 ca11
      COMPLEX*16 ca12
      COMPLEX*16 ca13
      COMPLEX*16 ca20
      COMPLEX*16 ca21
      COMPLEX*16 ca22
      COMPLEX*16 ca23
      COMPLEX*16 ca30
      COMPLEX*16 ca31
      COMPLEX*16 ca32
      COMPLEX*16 ca33

C   Define the sme coefficients

      ca00 = (0D0,0D0)
      ca01 = (0.05D0,-0.43D0)
      ca02 = (0D0,0D0)
      ca03 = (0D0,0D0)
      ca10 = (0.05D0,0.43D0)
      ca11 = (0D0,0D0)
      ca12 = (-1.88D0,0.72D0)
      ca13 = (0D0,0D0)
      ca20 = (0D0,0.0D0)
      ca21 = (-1.88D0,-0.72D0)
      ca22 = (0D0,0D0)
      ca23 = (0D0,0D0)
      ca30 = (0D0,0.0D0)
      ca31 = (0D0,0.0D0)
      ca32 = (0D0,0.0D0)
      ca33 = (0D0,0D0)

C   Define some tmp variables

      tmp12=ca00*V3(3)+ca01*V3(4)+ca02*V3(5)+ca03*V3(6)
      tmp13=ca10*V3(3)+ca11*V3(4)+ca12*V3(5)+ca13*V3(6)
      tmp14=ca20*V3(3)+ca21*V3(4)+ca22*V3(5)+ca23*V3(6)
      tmp15=ca30*V3(3)+ca31*V3(4)+ca32*V3(5)+ca33*V3(6)

      F1(3)= COUP*(-0.5D0)*CI*((-tmp12+tmp15)*F2(5)+(tmp13+CI*tmp14)*F2(6))
      F1(4)= COUP*(-0.5D0)*CI*((tmp13-CI*tmp14)*F2(5)+(-tmp12-tmp15)*F2(6))
      F1(5)= COUP*(-0.5D0)*CI*((tmp12+tmp15)*F2(3)+(tmp13+CI*tmp14)*F2(4))
      F1(6)= COUP*(-0.5D0)*CI*((tmp13-CI*tmp14)*F2(3)+(tmp12-tmp15)*F2(4))

      END










