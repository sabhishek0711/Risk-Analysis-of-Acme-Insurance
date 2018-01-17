function [projv2,projv1] = point_to_line(pt, v1, v2)
      a = v1 - v2;
      b = pt - v2;
      c = pt - v1;
      thetav2 = acosd(norm(dot(a,b)) / (norm(a)*norm(b)));
      thetav1 = acosd(norm(dot(a,c)) / (norm(a)*norm(c)));
      
      projv1 = norm(dot(a,b))/norm(a);
      projv2 = norm(dot(a,c))/norm(a);