% DC motor speed Control With neural network

clc;
clear;
close all;
alpha=0.01;
Ra=1.2;
La=1.4e-3;
Ka=.055;
Kt=Ka;
J=.0005;
B=.01*J;
itNN=180;
n=itNN-1;
Neuron=12;    %Number of neurons in hidden layers
NumInt=7;     %Number of inputs

% Initial Conditions
dt=0.001;
Omega1=zeros(1,itNN);
Theta1=zeros(1,itNN);
Ia=zeros(1,itNN);
% input 
ref2=randi([20 20],1,30);
ref22=randi([30 30],1,30);
ref33=randi([40 40],1,30);
ref34=randi([50 50],1,30);
ref35=randi([60 60],1,30);
ref36=randi([25 25],1,30);
ref3=[ref2 ref22 ref33 ref34 ref35 ref36];

%% reference model

%  reference=ref3;

reference=randi([20 20],1,itNN);

%% Initialize DRNN Controller

        
% Weights
W_w.w1=[];        % Weight From inputs to  Hidden layer1
W_w.w2=[];        % Weight From Hidden layer1 to Hidden layer2
W_w.w3=[];        % Weight From Hidden layer2 to Hidden layer3
W_w.w4=[];        % Weight From Hidden layer3 to output node




Ww=repmat(W_w,itNN ,1);

for k=1:itNN
    Ww(k).w1=unifrnd(0,1,NumInt,Neuron);
    Ww(k).w2=unifrnd(0,1,Neuron,Neuron);
    Ww(k).w3=unifrnd(0,1,Neuron,Neuron);
    Ww(k).w4=unifrnd(0,1,1,Neuron);
    Ww(k).z=zeros(1,Neuron);

end

E=zeros(1,itNN);                              %Error between Omega and input signal
E1=zeros(1,itNN);                             % observing and reduce Error
E2=zeros(1,itNN);                             % Error between Ref.model and RNN output 
u=zeros(1,itNN);                                %NN OutPut
Input_of_Hidden_layer1=zeros(1,itNN);
Delta=zeros(1,itNN);


for it=1:5000
for i=4:n

  Input_of_Hidden_layer1 =Ww(i).w1'*[ref3(i) E1(i) E1(i-1) Omega1(i) Omega1(i-1) u(i-1) u(i)]'+1;


    output_ofHidden_layer1 =  tanh(Input_of_Hidden_layer1);
     
     Input_of_Hidden_layer2 = (Ww(i).w2*output_ofHidden_layer1)+1;
     
     output_ofHidden_layer2 =  tanh(Input_of_Hidden_layer2);
     
     Input_of_Hidden_layer3 = (Ww(i).w3*output_ofHidden_layer2)+1;
     
     output_ofHidden_layer3 =  tanh(Input_of_Hidden_layer3);
     
     Input_of_output_Node =   (Ww(i).w4*output_ofHidden_layer3)+1;
     
     u(i) =(Input_of_output_Node); % out put of output layer(linear)
    
    Theta1(i+1)=Theta1(i)+dt*Omega1(i);
    
    Omega1(i+1)=Omega1(i)+dt*(-B*Omega1(i)+Kt*Ia(i))/J;
     
    Ia(i+1)=Ia(i)+dt*(-Ka*Omega1(i)-Ra*Ia(i)+u(i))/La;

    E1(i+1)=(ref3(i+1)-Omega1(i+1));
    E2(i)=(reference(i+1)-Omega1(i+1));
    E(i)=(E2(i)-u(i));

    error_of_hidden_layer3=Ww(i).w4'*E2(i+1);
   
   Delta3=(Input_of_Hidden_layer3>0).*error_of_hidden_layer3;
    
    error_of_hidden_layer2=Ww(i).w3'*Delta3;
    Delta2=(Input_of_Hidden_layer2>0).*error_of_hidden_layer2;
    
    error_of_hidden_layer1=Ww(i).w2'*Delta2;
    Delta1=(Input_of_Hidden_layer1>0).*error_of_hidden_layer1;
    
    adjustment_of_W4= alpha*E2(i+1) * output_ofHidden_layer3';
    adjustment_of_W3= alpha*Delta3 * output_ofHidden_layer2';
    adjustment_of_W2= alpha*Delta2* output_ofHidden_layer1';
    adjustment_of_W1= alpha*Delta1*[ref3(i) E1(i) E1(i-1) Omega1(i) Omega1(i-1) u(i-1) u(i)];
    
      %Transpose adjusted Weights
    adjustment_of_W1=adjustment_of_W1';
    adjustment_of_W2=adjustment_of_W2';
    adjustment_of_W3=adjustment_of_W3';
    
    Ww(i).w1=Ww(i).w1+adjustment_of_W1;
    Ww(i).w2=Ww(i).w2+adjustment_of_W2;
    Ww(i).w3=Ww(i).w3+adjustment_of_W3;
    Ww(i).w4=Ww(i).w4+adjustment_of_W4;


end
end

%% Results

 % RNN Elman Types OutPuts
  figure;
  plot(reference,'LineWidth',2)   
  hold on;
  plot(Omega1,'r','LineWidth',1.2)
  
  legend('reference','system output')
  title('NN Controler')
  hold off
