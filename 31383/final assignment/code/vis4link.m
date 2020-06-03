function vis4link(M,svinkel,inc)
% Funktionen vis4link viser en grafisk 3D animation af
% simuleringerne med LPT-robotten.
%
% Syntax :
% 	vis4link(M,svinkel,inc)
%
%	M en matrix der beskriver robottens bevægelser,
%	  den genereres af simulink blokken Gem til vis4link. 
%
%	svinkel vælger synsvinkel
%	 svinkel=1 : Normal
%	 svinkel=2 : Forfra	
%	 svinkel=3 : Bagfra
%	 svinkel=4 : Oven fra
%
%	inc bestemmer hvor mange af de simulerede punkter der 
%	skal vises inc=2 viser hvert andet punkt osv.
%
% Hvis det animerede tidsforløb ønskes korekt, skal simulationen
% der ligger til grund for animationen køres med konstante 
% integrationsstep.
%
% efter kørsel udskrives 2 tal på workspacen det første er den
% tid animationen har taget, det andet tal er den værdi af inc der
% gør at animationen køre i sand tid.
%
% Projektopgave i Robotteknik (41625) E2005

global d1

% Her sættes parametre til figure vinduet
fh=figure;				
set(fh,'Name','Vis4Link');		
set(fh,'Numbertitle','off');		
set(fh,'Position',[0 0 800 600]);	
set(fh,'Color',[1 1 1]);		

% Her sættes axis parametrene
xmin=-1;
xmax=3;
ymin=-2;
ymax=2.;
zmin=0.;
zmax=2.;
axis([xmin xmax ymin ymax zmin zmax]);
set(gca,'Visible','on');
set(gca,'DrawMode','fast');
set(gca,'GridLineStyle','-')
hold on 

% Her sættes synsvinkel
if svinkel==1;view([95 20]);end;	%normal
if svinkel==2,view([175 20]);end;	%for
if svinkel==3,view([10 20]);end;	%bag
if svinkel==4,view([90 90]);end;	%oven
if svinkel==5,view([90 0]);end;		%siden

% Her sættes labels på akserne
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');

% Her tegnes de 3 hvide vægge
fill3([xmax xmax xmin xmin],[ymin ymax ymax ymin],[zmin zmin zmin zmin],'g')
fill3([xmin xmin xmin xmin],[ymin ymax ymax ymin],[zmin zmin zmax zmax],'g')
fill3([xmax xmin xmin xmax],[ymin ymin ymin ymin],[zmin zmin zmax zmax],'g')

% Her tegnes robotsoklen
b1=0.5;	 % Bredden af toppen
ll1=1.0; % Længden af toppen
b2=0.6;	 % Bredden af bunden
ll2=1.2; % Længden af bunden
b3=0.8;	 % Højden af soklen

fill3([ll1/2 -ll1/2 -ll2/2 ll2/2],[-b1/2 -b1/2 -b2/2 -b2/2],[b3 b3 0 0],'y')
fill3([ll2/2 ll1/2 ll1/2 ll2/2],[b2/2 b1/2 -b1/2 -b2/2],[0 b3 b3 0],'y')
fill3([-ll2/2 -ll1/2 ll1/2 ll2/2],[b2/2 b1/2 b1/2 b2/2],[0 b3 b3 0],'y')
fill3([-ll2/2 -ll1/2 -ll1/2 -ll2/2],[-b2/2 -b1/2 b1/2 b2/2],[0 b3 b3 0],'y')
fill3([-ll1/2 -ll1/2 ll1/2 ll1/2],[b1/2 -b1/2 -b1/2 b1/2],[b3 b3 b3 b3],'y')
fill3([-ll2/2 -ll2/2 ll2/2 ll2/2],[b2/2 -b2/2 -b2/2 b2/2],[0 0 0 0],'y')

% Tegn link 1
line([0 0],[0 0], [b3 d1]);

% Her får Alto-robotten sit navn
text(0, b2/2, b3*2/3,'ALTO-robot','FontSize',14);

% Her skrives MEK.
text(xmin,ymin,zmax*2/3,'MEK*DTU','FontSize',14);

% Her defineres de robot links og et handle til den gemmes i; lh
lh=line([0 M(1,1);M(1,1) M(1,4)],[0 M(1,2);M(1,2) M(1,5)],[d1 M(1,3);M(1,3) M(1,6)],'EraseMode','xor');

% % Her printes simuleringstiden.
th=text(xmin,ymax,zmax,num2str(M(1,10)));

% Her sættes liniebredde og farve for de 2 links.
set(lh(1),'Color','b')
set(lh(2),'Color','b')


% I nedenstående løkke foretages selve animationen ved at de 2 definerede
% liniers kordinater bliver ændret løbende
tic;
for it=2:inc:length(M(:,9))
 set(lh(1),'xdata',[0 M(it,1)],'ydata',[0 M(it,2)],'zdata',[d1 M(it,3)]);
 set(lh(2),'xdata',[M(it,1) M(it,4)],'ydata',[M(it,2) M(it,5)],'zdata',[M(it,3) M(it,6)]);
 set(th,'string',num2str(M(it,10)));
 drawnow
 % Her plottes TCP' banekurve.
plot3(M(it,4),M(it,5),M(it,6),'-b'); 
end

animations_tid=toc
eqinc=inc*toc/M(length(M(:,10)),7)

plot3(M(:,4),M(:,5),M(:,6),'-r');
