function r = shortest_distance(pt,fault_length,fault_pnt_pos,pos_xy,SRL)

     v1 = [pos_xy(1) pos_xy(2)];
     v2 = [pos_xy(3) pos_xy(4)];
     a = v1 - v2;
     b = pt - v2;
     c = pt - v1;
     X1 = [pt;v1];
     X2 = [pt;v2];
     r0 = norm(cross([a,0],[b,0]))/norm(a);
     r1 = min([pdist(X1,'euclidean') pdist(X2,'euclidean')]);
     r2 = max([pdist(X1,'euclidean') pdist(X2,'euclidean')]);

 if fault_pnt_pos == 1
     
     d = sqrt(r1^2 - r0^2);
     r = sqrt((rand()*(fault_length - SRL) + d)^2 + r0^2);
     
 else
     
     L1 = sqrt(r1^2 - r0^2);
     L2 = sqrt(r2^2 - r0^2);
     
     if SRL > L1
         
         x = rand()*(fault_length - SRL);
         
         if x <= L1
             r = r0;
         else
             r = sqrt((x - L1)^2 + r0^2);
         end
     else
         
         x = rand()*(fault_length - SRL);
         
         if x < (L1 - SRL)
             
             r = sqrt((L1 - x - SRL)^2 + (r0)^2);
             
         elseif x>= (L1 - SRL) && x<= L1
             r = r0;
         else
             r = sqrt((x - L1)^2 + (r0)^2);
         end
         
     end
     
 end