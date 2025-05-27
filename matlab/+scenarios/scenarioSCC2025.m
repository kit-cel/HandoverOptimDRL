function params = scenarioSCC2025(params, ueData, latitude, longitude, maxNumSamples)
    % SCENARIO FILE FOR THE SCC 2025
    %   - Dataset generation
    % see also launcherFiles.launcherSCC2025

    %% General %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    slotsPerChunk = 50;             
    numSamples    = length(ueData.Latitude); 
    numPositions  = slotsPerChunk * floor(min(numSamples, maxNumSamples) / slotsPerChunk);

    params.time.slotDuration                = 1e-3;
    params.time.slotsPerChunk               = slotsPerChunk;
    params.time.numberOfChunks              = numPositions / slotsPerChunk;
    params.time.timeBetweenChunksInSlots    = 0;
    params.time.setSegmentLengthToOne       = true;

    params.carrierDL.centerFrequencyGHz             = 2.1; 
    params.transmissionParameters.DL.bandwidthHz    = 10e6;

    params.cellAssociationStrategy = parameters.setting.CellAssociationStrategy.maxReceivePower;

    params.postprocessor = simulation.postprocessing.LiteWithNetworkPP;
    params.smallScaleParameters.verbosityLevel = 0;

    params.save.losMapUEAnt             = true;
    params.save.isIndoor                = true;
    params.save.macroscopicFadingUEAnt  = true;
    params.save.antennaGain             = true;
    params.save.receivePower            = true;
    params.save.pathlossTableUEAnt      = true;

    %% Region of Interest %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    origin                          = [mean(latitude), mean(longitude), 0];
    [xMinMax, yMinMax]              = latlon2local(latitude, longitude, 0, origin);
    params.regionOfInterest.xSpan   = xMinMax(2) - xMinMax(1);
    params.regionOfInterest.ySpan   = yMinMax(2) - yMinMax(1);
    params.regionOfInterest.zSpan   = 75;

    %% Open Street Map City %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    osmCity                     = parameters.city.OpenStreetMap;
    osmCity.latitude            = latitude;
    osmCity.longitude           = longitude;
    osmCity.streetWidth         = 10;
    osmCity.minBuildingHeight   = 5;
    osmCity.maxBuildingHeight   = 30;
    osmCity.makeStreets         = true;
    osmCity.wallLossdB          = 3;
    osmCity.saveFile            = 'dataFiles/blockages/openStreetMap.json';
    osmCity.osmSaveDir          = 'dataFiles/blockages/openStreetMap.osm';
    params.cityParameters('osm') = osmCity;

    %% Pathloss Model Container / Ray Tracing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    indoor	= parameters.setting.Indoor.indoor;
    outdoor	= parameters.setting.Indoor.outdoor;
    LOS     = parameters.setting.Los.LOS;
    NLOS	= parameters.setting.Los.NLOS;
    macro = parameters.setting.BaseStationType.macro;

    params.pathlossModelContainer.modelMap{macro,	indoor,     LOS}    = parameters.pathlossParameters.UrbanMacro5G;
    params.pathlossModelContainer.modelMap{macro,	indoor,     NLOS}   = parameters.pathlossParameters.UrbanMacro5G;
    params.pathlossModelContainer.modelMap{macro,	outdoor,    LOS}    = parameters.pathlossParameters.UrbanMacro5G;
    params.pathlossModelContainer.modelMap{macro,	outdoor,	NLOS}   = parameters.pathlossParameters.UrbanMacro5G;
    params.pathlossModelContainer.modelMap{macro,	outdoor,	LOS}.isLos = true;
    params.pathlossModelContainer.modelMap{macro,	outdoor,	NLOS}.isLos = false;

    %% Shadow Fading %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params.shadowFading.on                  = true;
    params.shadowFading.resolution	        = 5;
    params.shadowFading.mapCorr             = 0.5;         
    params.shadowFading.meanSFV             = 0;           
    params.shadowFading.stdDevSFV	        = 1;           
    params.shadowFading.decorrDist	        = 20 * log(2);

    %% Base Stations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    bsLatitude      = [49.01089,    49.01009,   49.01024,   49.0053,    49.00534];
    bsLongitude     = [8.37934,     8.39415,    8.38630,    8.3830,     8.39026]; 
    azimuthList     = [250,         290,        270,        90,         90]; 
    elevationList   = [100,         100,        130,        100,        90];
    bsHeights       = [30,          30,         30,         30,         30];
    [antX, antY] = latlon2local(bsLatitude, bsLongitude, bsHeights, origin);

    for i = 1:length(bsLatitude)
        antenna = parameters.basestation.antennas.ThreeSector;
        antenna.nTX          = 4;
        antenna.azimuth      = azimuthList(i);
        antenna.elevation    = elevationList(i);
        antenna.technology   = parameters.setting.NetworkElementTechnology.NRMN_5G;
        antenna.precoderAnalogType	= parameters.setting.PrecoderAnalogType.none;

        baseStation                             = parameters.basestation.PredefinedPositions;
        baseStation.antenna.baseStationType     = parameters.setting.BaseStationType.macro;
        baseStation.positions	                = [antX(i); antY(i)];
        baseStation.nSectors                    = 1;
        baseStation.antenna	                    = antenna;
        baseStation.antenna.height              = bsHeights(i);
        
        params.baseStationParameters(strcat('BaseStation', num2str(i))) = baseStation;
    end

    %% Users %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    userHeight = 1.5;
    zPositions = userHeight * ones(1, numPositions);
    [xPositions, yPositions]    = latlon2local(ueData.Latitude(1:numPositions), ueData.Longitude(1:numPositions), zPositions, origin);
    
    user                            = parameters.user.PredefinedPositions;
    user.channelModel               = parameters.setting.ChannelModel.Rayleigh;
    user.positions                  = [0; 0; 0]; % will be overwritten
    user.movement                   = parameters.user.movement.Predefined;
    user.movement.positionList      = [xPositions; yPositions; zPositions];
    user.nRX                        = 1;
    user.nTX                        = 1;
    user.rxNoiseFiguredB            = 7;
    user.channelModel               = parameters.setting.ChannelModel.Rayleigh;
    user.indoorDecision             = parameters.indoorDecision.Geometry;
    user.losDecision                = parameters.losDecision.Geometry;
    user.technology                 = parameters.setting.NetworkElementTechnology.NRMN_5G;

    params.userParameters('users')  = user;
end
