// ----------------------------------------------------------------------------
// nexus | SquareFiberSteppingAction.cc
//
// This class stops tracking a particle when it meets specific conditions.
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#include "SquareFiberSteppingAction.h"
#include "FactoryBase.h"
#include "G4VProcess.hh"

#include <G4Step.hh>
#include <G4Track.hh>
#include <G4OpticalPhoton.hh>

using namespace nexus;

REGISTER_CLASS(SquareFiberSteppingAction, G4UserSteppingAction)

SquareFiberSteppingAction::SquareFiberSteppingAction(): G4UserSteppingAction()
{
}

SquareFiberSteppingAction::~SquareFiberSteppingAction()
{
}

void SquareFiberSteppingAction::UserSteppingAction(const G4Step* step)
{
  G4Track* track = step->GetTrack();
  G4ParticleDefinition* particleType = track->GetDefinition();

  // Specify the particle type and conditions under which tracking should stop
  if (particleType == G4OpticalPhoton::Definition()) {
    G4String volumeName = step->GetPreStepPoint()->GetTouchable()->GetVolume()->GetName();
    G4String processName = step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();

    if (volumeName == "TPB_Fiber" && track->GetParentID() == 0 && processName == "OpWLS") {
      track->SetTrackStatus(G4TrackStatus::fStopAndKill);

      G4cout << "G4OpticalPhoton::Definition(): ---> " << G4OpticalPhoton::Definition() << G4endl;
      G4cout << "particleType = " << particleType << G4endl;
      G4cout << "volumeName = " << volumeName << G4endl;
      G4cout << "processName = " << processName << G4endl;
      G4cout << "SquareFiberSteppingAction Fiber_TPB" << G4endl;
    }
  }

  return;
}
