#ifndef SQUARE_OPTICAL_FIBER_HH
#define SQUARE_OPTICAL_FIBER_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "GeometryBase.h"
#include <random>
#include "Randomize.hh"
#include "G4MultiUnion.hh"
#include "G4GenericMessenger.hh"
#include "G4VSensitiveDetector.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4Step.hh"
#include "G4VSensitiveDetector.hh"


namespace nexus {

    class SquareOpticalFiber : public GeometryBase {

    mutable std::mt19937 gen_; // move the generator to the class scope and make it mutable
    mutable std::uniform_real_distribution<double> z_dist_; // make the distribution mutable as well

    public:
        G4int seed_ = 42;
        SquareOpticalFiber();
        ~SquareOpticalFiber();
        // void UpdateELGapLimits();
        std::pair<G4double, G4double> UpdateELGapLimits();
        void Construct();
        G4GenericMessenger *msg_;
        /// Return vertex within region <region> of the chamber
        G4ThreeVector GenerateVertex(const G4String& region) const;

        // // my addition - trying something 30.4.23
        // void SetSiPMOutputPath(const G4String& path);
        // void SetTPBOutputPath(const G4String& path);
        // void SetNewValue(G4UIcommand* command, G4String newValue);
        // SquareFiberSD* squareFiberSD_;
        // G4GenericMessenger *msgSD_;
        // G4UIcmdWithAString* sipmPathCmd_;
        // G4UIcmdWithAString* tpbPathCmd_;


        G4double holderThickness_;
        G4double TPBThickness_;

        // Controlled from macro
        G4ThreeVector specific_vertex_;
        G4double ELGap_;
        G4double pitch_;
        G4double distanceFiberHolder_;
        G4double distanceAnodeHolder_;

    };
}

#endif