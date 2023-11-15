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


// original
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

// //second one
// void SquareFiberTrackingAction::PreUserTrackingAction(const G4Track* track) {
//     if (track->GetParentID() != 0) {  // Check only for secondary particles
//         const G4VProcess* creator = track->GetCreatorProcess();
//         if (creator && creator->GetProcessName() == "OpWLS") {
//             MyTrackInfo* info = dynamic_cast<MyTrackInfo*>(track->GetUserInformation());
//             G4cout << "info = " << info << G4endl;
//             if (!info) {
//                 info = new MyTrackInfo();
//                 track->SetUserInformation(info);
//             }
//             info->SetParentHasOpWLS(true);
//             G4cout << "OpWLS process detected for track ID: " << track->GetTrackID() << G4endl;
//         }
//     }
// }









void SquareFiberTrackingAction::PostUserTrackingAction(const G4Track* track) {
    // Optional: Implement this method if needed
}