function data = filterGPXtracks(data, latitude, longitude)
    % FILTER GPX TRACKS
    % input:
    %   data:[]table table with data
    %   latitude:[1x2]double Latitudes of the selected area
    %   longitude:[1x2]double Longitudes of the selected area
    %
    % output:
    %   msTable:[]table table with the filtered measurement data

    idxs = data.Latitude > latitude(1);
    data.Latitude = data.Latitude(idxs);
    data.Longitude = data.Longitude(idxs);

    idxs = data.Latitude < latitude(2);
    data.Latitude = data.Latitude(idxs);
    data.Longitude = data.Longitude(idxs);

    idxs = data.Longitude > longitude(1);
    data.Latitude = data.Latitude(idxs);
    data.Longitude = data.Longitude(idxs);

    idxs = data.Longitude < longitude(2);
    data.Latitude = data.Latitude(idxs);
    data.Longitude = data.Longitude(idxs);
end