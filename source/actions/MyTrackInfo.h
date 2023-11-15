#ifndef MY_TRACK_INFO_HH
#define MY_TRACK_INFO_HH

#include "G4VUserTrackInformation.hh"
#include "G4String.hh"

namespace nexus {

    class MyTrackInfo : public G4VUserTrackInformation {
    public:
        MyTrackInfo() : parentHasOpWLS(false) {}
        virtual ~MyTrackInfo() {}

        void SetParentHasOpWLS(bool value) { parentHasOpWLS = value; }
        bool GetParentHasOpWLS() const { return parentHasOpWLS; }

    private:
        bool parentHasOpWLS;
    };
}

#endif // MY_TRACK_INFO_HH