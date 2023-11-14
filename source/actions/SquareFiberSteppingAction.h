// ----------------------------------------------------------------------------
// nexus | SquareFiberSteppingAction.h
//
// This class stops tracking a particle when it meets specific conditions.
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#ifndef SQUARE_FIBER_STEPPING_ACTION_H
#define SQUARE_FIBER_STEPPING_ACTION_H

#include <G4UserSteppingAction.hh>
#include <globals.hh>

namespace nexus {

  class SquareFiberSteppingAction: public G4UserSteppingAction
  {
  public:
    /// Constructor
    SquareFiberSteppingAction();
    /// Destructor
    virtual ~SquareFiberSteppingAction();

    virtual void UserSteppingAction(const G4Step*);

  };

} // namespace nexus

#endif
