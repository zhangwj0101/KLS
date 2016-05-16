[alpha,beta]  = ldamain('C:/NMTF/Train.data.wf',50);
csvwrite(strcat('alpha.pwz'),alpha);
csvwrite(strcat('beta.pwz'),beta);