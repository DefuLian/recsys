function d = distanceGPS(latitude1, longitude1, latitude2, longitude2)   
    R = 6371; % radius of earth in km        
    d = ((acosd(sind(latitude1)*sind(latitude2) + cosd(latitude1)*cosd(latitude2) .* cosd(longitude2-longitude1)) * pi)/180.0) * R * 1000;
end