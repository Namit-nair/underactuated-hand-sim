#!/usr/bin/python
try:
    from . import owl
except ImportError:
    import owl


class MoCap:

    def __init__(self, server="192.168.1.230"):

        self.server = server

        self.o = owl.Context()

        self.o.open(self.server)

        self.o.initialize("streaming=1")

        print("MoCap connected")


    def get_rigids(self):

        evt = self.o.nextEvent(1000000)

        if evt and evt.type_id == owl.Type.FRAME:

            if "rigids" in evt:

                rigid_list = []

                for r in evt.rigids:

                    if r.cond > 0:

                        rigid_data = {
                            "id": r.id,
                            "position": r.pose[:3],
                            "quaternion": r.pose[3:],
                            "cond": r.cond
                        }

                        rigid_list.append(rigid_data)

                return rigid_list

        return None


    def close(self):

        self.o.done()

        self.o.close()

        print("MoCap closed")
