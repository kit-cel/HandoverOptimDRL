"""Extract GPX data from SUMO simulation using TraCI."""

import os
import sys
import traci
import gpxpy
import gpxpy.gpx

SIM_LENGTH = 10_000
STEP = 0.01

TOTAL_STEPS = int(SIM_LENGTH / STEP)
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

UE_SPEED = 70


def main() -> int:
    """Main function to extract GPX data from SUMO simulation."""
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # TraCI command
    sumo_cmd = [
        "sumo",
        "-n",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.net.xml"),
        "-r",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "routes.rou.xml",
        ),
        "-a",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "vehicles.rou.xml"),
        "--step-length",
        f"{STEP}",
    ]
    traci.start(sumo_cmd)

    obj_positions = {}
    for _ in range(TOTAL_STEPS):
        traci.simulationStep()

        for veh_id in traci.vehicle.getIDList():
            position = traci.vehicle.getPosition(veh_id)
            lon, lat = traci.simulation.convertGeo(position[0], position[1])

            if veh_id not in obj_positions:  # Add vehicle if not already in dict
                obj_positions[veh_id] = []

            obj_positions[veh_id].append((lon, lat))

        for person_id in traci.person.getIDList():
            position = traci.person.getPosition(person_id)
            lon, lat = traci.simulation.convertGeo(position[0], position[1])

            if person_id not in obj_positions:  # Add person if not already in dict
                obj_positions[person_id] = []

            obj_positions[person_id].append((lon, lat))

    traci.close()

    # Create files
    for veh_id, positions in obj_positions.items():
        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)

        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        for lon, lat in positions:
            gpx_point = gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
            gpx_segment.points.append(gpx_point)

        filename = f"ue_{UE_SPEED}_{veh_id}_positions.gpx"
        with open(
            os.path.join(RESULT_DIR, f"{UE_SPEED}", filename), "w", encoding="utf-8"
        ) as f:
            f.write(gpx.to_xml())
        print(f"GPX-file saved for {veh_id}: {filename}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
