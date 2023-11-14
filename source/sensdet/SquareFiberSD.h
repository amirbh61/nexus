#ifndef SQUARE_FIBER_SD_HH
#define SQUARE_FIBER_SD_HH

#include "G4VSensitiveDetector.hh"
#include "G4String.hh"
#include "G4Step.hh"
#include "G4GenericMessenger.hh"

namespace nexus {

  class SquareFiberSD : public G4VSensitiveDetector {
  public:
    SquareFiberSD(G4String const& SD_name, G4String const& sipmOutputFileName,
                   G4String const& tpbOutputFileName);
    ~SquareFiberSD();


    G4bool ProcessHits(G4Step* step, G4TouchableHistory* history);

    void WritePositionToTextFile(std::ofstream&, double x, double y);


    // // Setters for output file paths
    void SetSipmPath(const G4String& path);
    void SetTpbPath(const G4String& path);

    G4GenericMessenger *msgSD_;
    std::ofstream sipmOutputFile_;
    std::ofstream tpbOutputFile_;
    G4String sipmOutputFileName_;
    G4String tpbOutputFileName_;
  };

}

#endif
