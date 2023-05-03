#include "G4OpticalPhoton.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4TouchableHistory.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include <fstream>
#include <sstream>
#include "SquareFiberSD.h"
#include "G4VProcess.hh"


namespace nexus{


SquareFiberSD::SquareFiberSD(const G4String& name): G4VSensitiveDetector(name), msgSD_(nullptr),
sipmOutputFile_("/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/SiPM_hits.txt"),
tpbOutputFile_("/media/amir/9C33-6BBD/NEXT_work/Geant4/nexus/TPB_hits.txt")
{
  // msgSD_ = new G4GenericMessenger(this, "/Geometry/SquareFiberSD/Output/", "Commands for setting output file paths.");
  // msgSD_->DeclareMethod("sipmPath", &SquareFiberSD::SetSiPMOutputFilePath, "Set the output file path for SiPM hits.");
  // msgSD_->DeclareMethod("tpbPath", &SquareFiberSD::SetTPBOutputFilePath, "Set the output file path for TPB hits.");
}



SquareFiberSD::~SquareFiberSD() {
  if (sipmOutputFile_.is_open()) {
    sipmOutputFile_.close();
  }
  if (tpbOutputFile_.is_open()) {
    tpbOutputFile_.close();
  }
}




G4bool SquareFiberSD::ProcessHits(G4Step* step, G4TouchableHistory*) {
  G4Material* material = step->GetPreStepPoint()->GetMaterial();
  // G4Material* material = step->GetPostStepPoint()->GetMaterial();

  std::string materialName = material->GetName();
  G4Track* track = step->GetTrack();

  // std::cout << "Material: " << materialName << std::endl;
  // std::cout << "track->GetParentID() = " << track->GetParentID() << std::endl;
  // std::cout << "track->GetCurrentStepNumber() = " << track->GetCurrentStepNumber() << std::endl;

  if (materialName == "TPB" &&
      track->GetParentID() == 0) {

    // std::cout << "TPB hit! #" << std::endl;
    G4ThreeVector position = step->GetPreStepPoint()->GetPosition();
    WritePositionToTextFile(tpbOutputFile_, position);

  }
  
  if (materialName == "G4_Si" &&
      track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() == "OpAbsorption"){

    G4ThreeVector position = step->GetPreStepPoint()->GetPosition();
    WritePositionToTextFile(sipmOutputFile_, position);

  }

  return true;
}

void SquareFiberSD::WritePositionToTextFile(std::ofstream& file, const G4ThreeVector& position) {
  if (file.is_open()) {
    file << position.x() << " " << position.y() << " " << position.z() << std::endl;
  } else {
    std::cerr << "Error: Unable to open output file" << std::endl;
  }
}



// Implementation of setter functions
void SquareFiberSD::SetSiPMOutputFilePath(const G4String& filePath) {
  sipmOutputFile_.open(filePath.c_str(), std::ios_base::app);
}


void SquareFiberSD::SetTPBOutputFilePath(const G4String& filePath) {
  tpbOutputFile_.open(filePath.c_str(), std::ios_base::app);
}


} // close namespace nexus