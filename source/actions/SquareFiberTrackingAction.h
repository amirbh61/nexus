// MyTrackingAction.hh
#ifndef SQUARE_FIBER_TRACKING_ACTION_HH
#define SQUARE_FIBER_TRACKING_ACTION_HH

#include "G4UserTrackingAction.hh"
#include "MyTrackInfo.h"

class G4Track;

namespace nexus {

    class SquareFiberTrackingAction : public G4UserTrackingAction {
    public:
        SquareFiberTrackingAction();
        virtual ~SquareFiberTrackingAction();

        virtual void PreUserTrackingAction(const G4Track*);
        virtual void PostUserTrackingAction(const G4Track*);
    };
}

#endif // SQUARE_FIBER_TRACKING_ACTION_HH
