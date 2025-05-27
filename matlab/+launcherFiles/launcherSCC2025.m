% LAUNCHER FILE FOR THE SCC 2025
%   - Dataset generation
% see also scenarios.scenarioSCC2025

close all;
clear;
clc;

simulationID = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));

% Parameters
latitude            = sort([49.004900, 49.011200]); 
longitude           = sort([8.377300,   8.395600]);
numTracks           = 100;
maxNumUserPositions = 1e4;
userSpeed           = 50;

fprintf("dataFiles/sumo/" + num2str(userSpeed) + "/ue_" + num2str(userSpeed) + "_" + num2str(i-1) + "_positions.gpx")
cnt = 0;
for i = 1:numTracks
    try
        % Read GPX file
        tracks = gpxread("dataFiles/sumo/" + num2str(userSpeed) + "/ue_" + num2str(userSpeed) + "_" + num2str(i-1) + "_positions.gpx");
        tracksLatitude = tracks.Latitude;
        tracksLongitude = tracks.Longitude;
        fprintf("Length of track " + num2str(i) + ": "+ num2str(length(tracks))+'\n');

        numPoints = length(tracksLatitude);
        distances = zeros(numPoints-1, 1);
        for k = 1:numPoints-1
            distances(k) = distance(tracksLatitude(k), tracksLongitude(k), tracksLatitude(k+1), tracksLongitude(k+1), wgs84Ellipsoid("m"));
        end
        
        avgSpeedKmh = sum(distances) / 1000 / (numPoints - 1) * 0.01 / 3600;
        fprintf("Average Speed: "+ num2str(avgSpeedKmh)+ "km/h\n");

        ueData = struct();
        ueData.Latitude = tracksLatitude;
        ueData.Longitude = tracksLongitude;

        ueData = parser.filterGPXtracks(ueData, latitude, longitude);

        fprintf("Length of tracks: "+num2str(size(ueData.Latitude)));
        scatter(ueData.Longitude, ueData.Latitude);

        % Launch simulations
        simMode = parameters.setting.SimulationType.parallel;
        result = simulate(@(params)scenarios.scenarioSCC2025(params, ueData, latitude, longitude, maxNumUserPositions), simMode);

        rsrp = result.additional.receivePowerdB;
        sinr = result.widebandSinrdBAllAntennas;
        bsidxs = result.userToBSassignment;

        save("results/data/rsrp/ue" + num2str(round(avgSpeedKmh)) + "kmh_rsrp_" + num2str(cnt) + ".mat", "rsrp");
        save("results/data/sinr/ue" + num2str(round(avgSpeedKmh)) + "kmh_sinr_" + num2str(cnt) + ".mat", "sinr");
        save("results/data/bsidxs/ue" + num2str(round(avgSpeedKmh)) + "kmh_bsidxs_" + num2str(cnt) + ".mat", "bsidxs");

        cnt = cnt + 1;
    catch
        warning("File not found: obj_" + num2str(i-1) + "_positions.gpx")
    end
end