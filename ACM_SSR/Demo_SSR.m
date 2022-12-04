%% A single-scale modulated active contour model for fast image segmentation
% Author: Guirong Weng(ÎÌ¹ðÈÙ); Yu Lei; Chengxin Yang 
% School of Mechanical and Electric Engineering, Soochow University, Suzhou 215021, China)
% All rights researved by Guirong Weng, Yu Lei, Chengxin Yang, who formulated the model, designed
% and implemented the algorithm in the above paper. Mar 29, 2021
% PR(Pattern Recognition)
% E-mail: wgr@suda.edu.cn

%  single-scale retinex model theory:
%  r_simga = i(x) - G_sigma*i;                                    in Eq.(4)
%  i = logI:image observed I, G_sigma is Gaussian kernel; * is convolution;
%  Difference of single-scale retinex: e = r_simga1 - r_simga2;  in Eq.(13)
%% -------------------------------------------------
clc; clear; close all;
imgID = 14 ;                          % imgID = 1 - 15
Img1 = imread([num2str(imgID),'.bmp']);
Img = double(Img1(:,:,1)); 
c0 = 1;                             % constant
initialLSF = ones(size(Img(:,:,1))).*c0; 

% The reference values of experimental parameters
[c,alfa,k,w,sigma1,sigma2,initialLSF] ...
    = SSR_Switch(c0,imgID,initialLSF);

Img = log(1+Img/255);               % i=log(I),rescale the image intensities
fmin  = min(Img(:));
fmax  = max(Img(:));
Img = 255*(Img-fmin)/(fmax-fmin);	% Normalize Img to the range [0,255]
u = initialLSF;

% -- constant parameter setting
timestep = 1;                       % constant
epsilon = 1;                        % constant
iterNum = 150;                      % Maximum number of iterations
eta = 20;                           % constant
tau = 7;                            % constant

A = fspecial('average',k);          % Create predefined filter,in Eq.(22)

% -- single-scale retinex model 
G1 = fspecial('gaussian', w, sigma1);                    % Gaussian kernel
G2 = fspecial('gaussian', w, sigma2);                    % Gaussian kernel

DSSR = G1 - G2;                 % Difference of single-scale retinex kernel
e = imfilter(Img,DSSR,'replicate');                      % in Eq.(13)
% r_G1 = imfilter(Img,G1,'replicate');
% r_G2 = imfilter(Img,G2,'replicate');
% e = r_G1 - r_G2;                                       % in Eq.(13)

% ------- Data driven function ------------------------
ex = - c*alfa*tsign(e/tau);  % constant tau=7,in Eq.(19),excluding DiracU

figure;imagesc(Img1);colormap(gray);hold on;axis off; axis equal;
contour(initialLSF,[0 0],'g','linewidth',2);

%% -----start level set evolution-----

for n = 1:iterNum
    u1 = u;              % Delete this display code when calculating time
    
    % In the level set model or active contour model,
    % the code in this paper is the simplest iterative operation.
    % *******************************************************************
    DiracU = (epsilon/pi)./(epsilon^2.+u.^2);% Update Diranc function,in Eq.(17)
    u = u + timestep*ex.*DiracU;% Compute level set function evolution in Eq.(19)
    u = tsign(eta*u);                       % eta=20,regularization method, in Eq.(21)
    u = imfilter(u,A,'symmetric');          % Length term function,in Eq.(22)
   
    % *******************************************************************
    
    %% -- Delete display code when calculating time ---
     if mod(n,40) == 0    
        hold on;contour(u,[0 0],'c'); axis off,axis equal
        title([num2str(n),'  iterations']);pause(0.001);
     end
    %% -----------------------------------------   
     if abs(u - u1) < 0.001 % u1 is the contour of the previous iteration
        break  
     end
   
    
end

contour(u,[0 0],'r','linewidth',2);
iterNum=[num2str(n), ' iterations']; title(iterNum); 

%% -- end ----------------------------------------------------------------
  
function y = tsign(x)
    y = 2.^(x+1)./(1+2.^x)-1;                                  % in Eq.(20)
end

function [c,alfa,k,w,sigma1,sigma2,initialLSF]... 
                               = SSR_Switch(c0,imgID,initialLSF)
    % The reference values of experimental parameters are
    %     alfa=5;k=7;w=9;sigma1=0.5;sigma2=4.5;
    switch imgID
        case 1
            c=1;alfa=5;k=7;w=9;sigma1=0.5;sigma2=4.5;
            initialLSF(24:58,16:85) = -c0; 
        case 2
            c=1;alfa=5;k=7;w=9;sigma1=0.5;sigma2=4.5;
            initialLSF(23:55,23:68) = -c0;
        case 3
            c=-1;alfa=5;k=7;w=9;sigma1=0.5;sigma2=4.5;
            initialLSF(30:75,40:80) = -c0; 
        case 4     
            c=1;alfa=5;k=7;w=9;sigma1=0.5;sigma2=4.5;
            initialLSF(85:100,50:80) = -c0 ;
        case 5
            c=1;alfa=5;k=7;w=9;sigma1=0.5;sigma2=4.5;
            initialLSF(105:119,50:85) = -c0;
        case 6
            c=1;alfa=5;k=7;w=9;sigma1=0.5;sigma2=4.5;
            initialLSF(28:62,20:78)=-c0;
        case 7 
            c=1;alfa=5;k=7;w=9;sigma1=0.5;sigma2=4.5;
            initialLSF(25:45,40:65) = -c0;
        case 8 
            alfa=10; % tau in equation (19) cannot provide adaptive requirements
            % *** Research direction: adaptive alfa or tau parameters ***
            c=1;k=7;w=9;sigma1=0.5;sigma2=4.5;
            initialLSF(45:80,30:100) = -c0;      
        case 9  
            alfa=10; % tau in equation (19) cannot provide adaptive requirements
            c=1;k=7;w=9;sigma1=0.5;sigma2=4.5;
            initialLSF(55:70,55:70) = -c0;     
       case 10
            c=-1;alfa=5;k=7;w=9;sigma1=0.5;sigma2=4.5;
            initialLSF(40:75,35:90) = -c0;   
       case 11
            sigma1=2;w=17;
           % The bias field background is relatively strong,
           % increasing sigma is conducive to detecting the boundary signal
           c=1;alfa=5;k=7;sigma2=4.5;
           initialLSF(20:165,17:82)=-c0; 
        case 12
           alfa=3; % tau in equation (19) cannot provide adaptive requirements
           % The boundary is weak, the energy coefficient alfa is reduced
           c=1;k=7;w=9;sigma1=0.5;sigma2=4.5;
           initialLSF(15:45,20:105) = -c0;
        case 13  
           alfa=3;w=35; % The bias field background is relatively strong
           c=-1;k=7;sigma1=0.5;sigma2=4.5;
           initialLSF(120:180,100:340) = -c0;
        case 14
           alfa=3;w=35;k=9; 
           % The bias field background is relatively strong,
           c=-1;sigma1=0.5;sigma2=4.5;
           initialLSF(182:220,210:265) = -c0;
        case 15  
           alfa=3;% tau in equation (19) cannot provide adaptive requirements
           % The boundary is weak, the energy coefficient alfa is reduced
           c=-1;k=7;w=9;sigma1=0.5;sigma2=4.5;
           initialLSF(60:110,35:85) = -c0;
   end
end 

