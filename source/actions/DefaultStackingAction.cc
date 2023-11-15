// ----------------------------------------------------------------------------
// nexus | DefaultStackingAction.cc
//
// This class is an example of how to implement a stacking action, if needed.
// At the moment, it is not used in the NEXT simulations.
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------


#include "DefaultStackingAction.h"
#include "SquareFiberTrackingAction.h"
#include "FactoryBase.h"
#include "G4VProcess.hh"
#include "G4RunManager.hh"
#include "MyTrackInfo.h"

using namespace nexus;

REGISTER_CLASS(DefaultStackingAction, G4UserStackingAction)

DefaultStackingAction::DefaultStackingAction(): G4UserStackingAction()
{
}



DefaultStackingAction::~DefaultStackingAction()
{
}



G4ClassificationOfNewTrack
DefaultStackingAction::ClassifyNewTrack(const G4Track* /*track*/)
{
  return fUrgent;
}



// G4ClassificationOfNewTrack DefaultStackingAction::ClassifyNewTrack(const G4Track* track) {
//     MyTrackInfo* info = dynamic_cast<MyTrackInfo*>(track->GetUserInformation());
//     G4cout << info << G4endl;
//     if (info) {
//         G4cout << "Track has user info. OpWLS flag is " << info->GetParentHasOpWLS() << G4endl;
//         if (info->GetParentHasOpWLS()) {
//             return fKill;
//         }
//     } else {
//         G4cout << "Track has no user info" << G4endl;
//     }
//     return fUrgent;
// }




void DefaultStackingAction::NewStage()
{
  return;
}



void DefaultStackingAction::PrepareNewEvent()
{
  return;
}
