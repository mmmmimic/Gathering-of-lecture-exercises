% -------------------------------------------------------------------------
% laserscan.m
%
% Filen returnerer de synlige punkter for en laserscanner med en max. 
% afstand på 15m, set fra en given position og retning i et kendt område. 
% Resultatet returneres i form af et 2x361-array med henholdsvis vinkler og
% afstande.
%
% Skrevet af Christian Overvad, s031914, og Kasper Strange, s031924.
% 
% Sidst ændret 06-06-2006
%  
%
% Returns the visible points for a laser scanner with a max distance at 
% maxDistance meters, seen from a given position and direction in a 
% well-known area being (x,y,theta) the CURRENT GLOBAL POSITION OF THE
% LASER SCANNER. Scan width is restricted to 180 degrees.
% The result is returned in an 2xN-array with respectively angles and 
% distances.
% 
% Edited by: Rafael Olmos Galve s071150 at DTU, Denmark - rafa_olmos@hotmail.com
% Changes done: -Max laser scanner measurement distance as a parameter.
%               -Resolution is not restricted anymore, it is specified as
%               parameter.
%               -Changed theta acording to the convention so that theta=0
%               means pointing along x-axis
% Last change 12/05/2008.
% -------------------------------------------------------------------------

function scan = laserscan(x, y, theta, lines, maxDistance, resol)

%Bringing the theta to conventional form
theta = theta-pi/2;

% Number of scanning lines (deduced from scan width and resolution)
num_of_scanning_lines=floor(180/resol);

% -------------------------------------------------------------------------
% Function [M,N] = SIZE(X) for matrix X, returns the number of rows(M) and 
% columns(N) in X as separate output variables.
% -------------------------------------------------------------------------

[m, no_of_lines] = size(lines); % Totalt antal linier - Total num. of lines

% ---------------------------------------------------
% Præ-allokering af arrays - Preallocation of arrays
% ---------------------------------------------------

% Global system coordinates
x_start(1:no_of_lines) = 0;
y_start(1:no_of_lines) = 0;
x_end(1:no_of_lines) = 0;
y_end(1:no_of_lines) = 0;

% Laser scanner local system coordinates
trans_x_start(1:no_of_lines) = 0;
trans_y_start(1:no_of_lines) = 0;
trans_x_end(1:no_of_lines) = 0;
trans_y_end(1:no_of_lines) = 0;

b(1:no_of_lines) = 0;
a(1:no_of_lines) = 0;
scan(1:2,1:no_of_lines) = 0;

% --------------------------------------------------
% Start- og slutværdier for liniernes endepunkterne.
% Start and end values for the lines ending points.
% --------------------------------------------------

for i = 1:no_of_lines
    x_start(i) = lines(1,i);
    y_start(i) = lines(2,i);
    x_end(i)   = lines(3,i);
    y_end(i)   = lines(4,i);
    
% ---------------------------------------------------------------------
% Transformation af linierne til laserscannerens koordinatsystem.
% Transformation of the lines to the laser scanner's coordinate system.
% ---------------------------------------------------------------------   

    % Linierne konverteres til nye akser.
    % Lines are converted to the new coordinate system.
    trans_x_start(i)=(x_start(i)-x)*cos(theta) + (y_start(i)-y)*sin(theta);
    trans_y_start(i)=(y_start(i)-y)*cos(theta) - (x_start(i)-x)*sin(theta); 
    trans_x_end(i) = (x_end(i)-x)*cos(theta) + (y_end(i)-y)*sin(theta);
    trans_y_end(i) = (y_end(i)-y)*cos(theta) - (x_end(i)-x)*sin(theta);
    
    % Den mindste x-værdi sættes til x_start(i).
    % Starting and ending points are swapped if the x value of the 
    % starting point is bigger than the x value of the ending point.
    if trans_x_start(i) > trans_x_end(i)
        trans_x_temp = trans_x_start(i);
        trans_x_start(i) = trans_x_end(i);
        trans_x_end(i) = trans_x_temp;
        trans_y_temp = trans_y_start(i);
        trans_y_start(i) = trans_y_end(i);
        trans_y_end(i) = trans_y_temp;
    end;

% -----------------------------------------------------------------------
% Liniernes hældning og skæringspunkt med y-aksen udregnes (y = a + b*x).
% The lines slope and intersection point with the y-axis are worked out
% (y = a + b*x).
% -----------------------------------------------------------------------
    
    % Hvis linien ikke er lodret udregnes hældning og skæringspunkt med
    % y-aksen.
    % If the line isn't vertical, slope (b) and intersection point with the
    % y-axis (a) are worked out.
    if abs(trans_x_start(i) - trans_x_end(i)) >= 0.001
        b(i) = (trans_y_end(i) - trans_y_start(i))/(trans_x_end(i) - trans_x_start(i));
        a(i) = -b(i)*trans_x_start(i) + trans_y_start(i);
    end;
end;

% -------------------------------------------------------------------
% Konvertering af det, som laserscanneren ser til polære koordinater.
% Conversion of what the laser scanner sees to polar coordinates.
% -------------------------------------------------------------------

vertical_scanning_line=floor((90*pi/180+(pi/180*resol))/(pi/180*resol));

% For each laser scanner angle
for i = 1:num_of_scanning_lines
    % Laserscanerens maksimale måleafstand.
    % Laser scanner maximum measured distance (in meters).
    max_dist = maxDistance;
    % Gyldig afstand tættest på laserscanneren.
    % Closest distance from the laser scanner.
    min_dist = max_dist;
    % Aktuelle vinkel laserscanner.
    % Current laser scanner angle (from 0 to 180 degrees in steps of resol
    % degrees)
    phi = pi/180*i*resol - pi/180*resol;
    % x-koordinat for skæringpunktet mellem linien og den linie som
    % laserscanneren måler langs.
    % x and y coordinates for the intersection point between the line and 
    % the line along which the laser scanner is currently measuring.
    x_cross = max_dist;
    y_cross = max_dist;   
    
    % linierne gennemgås for at finde deres afstand til laserscanneren
    % Find distance from the lines to laser scanner in the current angle
    for j = 1:no_of_lines
        
        % Hvis laserscanneren måler lodret ud i luften...
        % If the laser scanner measures out vertically (at 90 degrees)
        if i == vertical_scanning_line
            
            % ... og den ser en lodret linie...
            % ... and it's a vertical line...
            if abs(trans_x_start(j) - trans_x_end(j)) < 0.001
                % just over the y axis
                if abs(trans_x_start(j)) < 0.001
                    if trans_y_start(j) < trans_y_end(j) && trans_y_start(j) > 0
                        x_cross = trans_x_start(j);
                        y_cross = trans_y_start(j);                    
                    elseif trans_y_end(j) < trans_y_start(j) && trans_y_end(j) > 0
                        x_cross = trans_x_end(j);
                        y_cross = trans_y_end(j); 
                    else
                        % the line is behind the robot
                        x_cross = max_dist; 
                        y_cross = max_dist; 
                    end;      
                end;
            else
                x_cross = 0;
                y_cross = a(j);              
            end;
            
        % Hvis der ikke måles lige ud i luften...
        % If it's not measuring out vertically...
        else
            
            % ...og linien er lodret...
            % ...and the line is vertical...
            if abs(trans_x_start(j) - trans_x_end(j)) < 0.001
                x_cross = trans_x_start(j);               
                
            % ... og linien er parallel og sammenliggende med den linie
            % som laserscanneren måler langs...
            % ...and the line has the same both a(j) and b(j) that the line
            % the laser scanner measures along...
            elseif abs(tan(phi) - b(j)) < 0.001 && abs(a(j)) < 0.001
                if i < vertical_scanning_line
                    x_cross = trans_x_start(j);
                else
                    x_cross = trans_x_end(j);
                end;             
                
            % ... og ikke er lodret eller parallel med den linie som 
            % laserscanneren måler langs...
            % ...and isn't vertical and hasn't the same both a(j) and b(j)
            % that the line the laser scanner measures along...
            else
                if abs(tan(phi) - b(j)) > 0.001
                    x_cross = a(j)/(tan(phi) - b(j));   
                else
                    % If the line is parallel and doesn't cross point (0,0)
                    x_cross = max_dist;
                end;
            end;
            y_cross = tan(phi)*x_cross;
        end;
     
        % Hvis x_cross eller y_cross ligger forkert, frasorteres punktet
        % If x_cross or y_cross lie erroneously the point is sorted out
        if (i < vertical_scanning_line && x_cross < 0) || (i > vertical_scanning_line && x_cross > 0) || (i == vertical_scanning_line && y_cross < 0)
            x_cross = max_dist;
            y_cross = max_dist;
        end;
        
        % Afstand til punktet
        % Distance to the point
        dist = sqrt(x_cross^2 + y_cross^2); 
        
        % Hvis afstanden er gyldigt og er den tætteste på laserscanneren 
        % anvendes den.
        % If the distance is valid and is the closest one to the laser
        % scanner it's used.
        if dist < min_dist && x_cross > (trans_x_start(j) - 0.001) && x_cross < (trans_x_end(j) + 0.001) && ( (y_cross > (trans_y_start(j) - 0.001) && y_cross < (trans_y_end(j)) + 0.001) || (y_cross > (trans_y_end(j) - 0.001) && y_cross < (trans_y_start(j) + 0.001)) )
            min_dist = dist;
        end;        
    end; % from for j
    
    % De polære koordinater returneres
    % The polar coordinates returned
    scan(1,i) = phi-pi/2;
    scan(2,i) = min_dist;
end;