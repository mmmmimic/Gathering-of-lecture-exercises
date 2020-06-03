function [] = orderPosenLaser( sckDemo, sckLaser )

pose=[]; absPose=[]; scanData=[]; success=0;

receiveMessage(sckDemo,0.02);
receiveMessage(sckLaser,0.02);

mssendraw(sckDemo,uint8(['eval $odox' 10]));
mssendraw(sckDemo,uint8(['eval $odoy' 10]));
mssendraw(sckDemo,uint8(['eval $odoth' 10]));
mssendraw(sckLaser,uint8(['scanget codex=TAG' 10]));