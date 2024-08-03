#include "SquareFiberTrackingAction.h"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "FactoryBase.h"

using namespace nexus;

REGISTER_CLASS(SquareFiberTrackingAction, G4UserTrackingAction)

SquareFiberTrackingAction::SquareFiberTrackingAction() : G4UserTrackingAction()
{
}

SquareFiberTrackingAction::~SquareFiberTrackingAction()
{
}


void SquareFiberTrackingAction::PreUserTrackingAction(const G4Track* track) {
    if (track->GetParentID() == 0) {
        track->SetUserInformation(new MyTrackInfo());
    } else {
        const G4VProcess* creator = track->GetCreatorProcess();
        if (creator && creator->GetProcessName() == "OpWLS") {
            MyTrackInfo* info = new MyTrackInfo();
            info->SetParentHasOpWLS(true);
            track->SetUserInformation(info);
            //G4cout << "Setting OpWLS flag for track " << track->GetTrackID() << G4endl;
        }
    }
}




void SquareFiberTrackingAction::PostUserTrackingAction(const G4Track* track) {
    // Optional: Implement this method if needed
}