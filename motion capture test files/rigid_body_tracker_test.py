# import sys
# sys.dont_write_bytecode = True  # Disable .pyc files for cleaner workspace
# from rigid_body_tracker  import MoCap

# mocap = MoCap()

# while True:

#     rigids = mocap.get_rigids()

#     if rigids:

#         for r in rigids:

#             print(r)


from rigid_body_tracker import MoCap
import csv
import time

mocap = MoCap()

with open('rigid_body_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'id', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw', 'cond'])

    while True:

        rigids = mocap.get_rigids()

        if rigids:

            for r in rigids:

                print(r)
                writer.writerow([
                    time.time(),
                    r['id'],
                    *r['position'],
                    *r['quaternion'],
                    r['cond']
                ])
                f.flush()