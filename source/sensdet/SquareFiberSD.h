#ifndef SQUARE_FIBER_SD_HH
#define SQUARE_FIBER_SD_HH

#include "G4VSensitiveDetector.hh"
#include "G4String.hh"
#include "G4Step.hh"
#include "G4GenericMessenger.hh"

namespace nexus {

  class SquareFiberSD : public G4VSensitiveDetector {
  public:
    SquareFiberSD(const G4String& name);
    virtual ~SquareFiberSD();

    virtual G4bool ProcessHits(G4Step* step, G4TouchableHistory* history);

    void WritePositionToTextFile(std::ofstream& file, const G4ThreeVector& position);

    // Setters for output file paths
    void SetSiPMOutputFilePath(const G4String& filePath);
    void SetTPBOutputFilePath(const G4String& filePath);

    G4GenericMessenger *msgSD_;
    std::ofstream sipmOutputFile_;
    std::ofstream tpbOutputFile_;
  };

}

#endif
