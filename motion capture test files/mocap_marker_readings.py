import owl

SERVER = "192.168.1.230"   # Change if needed

# Create context
context = owl.Context()

# Connect to server
context.open(SERVER)

# Initialize streaming
context.initialize("event.markers=1 event.rigids=0")

# Start streaming
context.streaming(1)

print("Waiting for marker data...\n")

try:
    while context.isOpen():

        # Get next event
        event = context.nextEvent(1000000)   # 1 second timeout

        if not event:
            continue

        # Check for frame data
        if event.type_id == owl.Type.FRAME:

            # Marker data available?
            if "markers" in event:

                for m in event.markers:

                    # cond > 0 means valid marker
                    if m.cond > 0:
                        print(
                            f"Marker {m.id}: "
                            f"x={m.x:.2f}, "
                            f"y={m.y:.2f}, "
                            f"z={m.z:.2f}"
                        )

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    context.done()
    context.close()