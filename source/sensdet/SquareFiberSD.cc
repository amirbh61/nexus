#include "G4OpticalPhoton.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4TouchableHistory.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "SquareFiberSD.h"
#include "G4VProcess.hh"
#include "G4Material.hh"
#include <fstream>
#include <sstream>
#include <cstdio>

#include <iomanip>  // for std::fixed and std::setprecision




namespace nexus{


SquareFiberSD::SquareFiberSD(G4String const& SD_name, G4String const& sipmOutputFileName,
 G4String const& tpbOutputFileName): G4VSensitiveDetector(SD_name),
 sipmOutputFileName_(sipmOutputFileName),
 tpbOutputFileName_(tpbOutputFileName)
{

  // Remove SiPM and TPB files, if exist from previous run 
  if (std::remove(sipmOutputFileName.c_str()) != 0) {
    std::cout << "Failed to delete SiPM output file." << std::endl;
  } else {
    std::cout << "SiPM output file deleted." << std::endl;
  }

  if (std::remove(tpbOutputFileName.c_str()) != 0) {
    std::cout << "Failed to delete SiPM output file." << std::endl;
  } else {
    std::cout << "SiPM output file deleted." << std::endl;
  }


  SetSipmPath(sipmOutputFileName);
  SetTpbPath(tpbOutputFileName);
}




SquareFiberSD::~SquareFiberSD() {
  if (sipmOutputFile_.is_open()) {
    sipmOutputFile_.close();
    std::cout << std::endl;
    std::cout << "SiPM output file :" << std::endl << sipmOutputFileName_ << std::endl << "Closed successfully." << std::endl;
    std::cout << std::endl;
    }

  if (tpbOutputFile_.is_open()) {
    tpbOutputFile_.close();
    std::cout << std::endl;
    std::cout << "TPB output file :" << std::endl << tpbOutputFileName_ << std::endl << "Closed successfully." << std::endl;
    std::cout << std::endl;
  }
}



// ORIGINAL VERSION

// G4bool SquareFiberSD::ProcessHits(G4Step* step, G4TouchableHistory*) {
//   G4Material* material = step->GetPreStepPoint()->GetMaterial();
//   //G4Material* material = step->GetPostStepPoint()->GetMaterial();

//   std::string materialName = material->GetName();

//   std::string volumeName = step->GetPreStepPoint()->GetTouchable()->GetVolume()->GetName();
//   G4Track* track = step->GetTrack();

//   // std::cout << "Material: " << materialName << std::endl;
//   // std::cout << "track->GetParentID() = " << track->GetParentID() << std::endl;
//   // std::cout << "track->GetCurrentStepNumber() = " << track->GetCurrentStepNumber() << std::endl;


//   // Record photons interacting in the Fiber TPB only
//   if (volumeName == "TPB_Fiber" && track->GetParentID() == 0 &&
//      track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() == "OpWLS" ) {
//       G4ThreeVector position = step->GetPostStepPoint()->GetPosition();
//       WritePositionToTextFile(tpbOutputFile_, position);
//   }

  
//   if (materialName == "G4_Si" &&
//       track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() == "OpAbsorption"){

//     G4ThreeVector position = step->GetPreStepPoint()->GetPosition();
//     WritePositionToTextFile(sipmOutputFile_, position);

//   }

//   return true;
// }




G4bool SquareFiberSD::ProcessHits(G4Step* step, G4TouchableHistory*) {
  // Get the name of the volume where interaction happens
  std::string volumeName = step->GetPreStepPoint()->GetTouchable()->GetVolume()->GetName();
  G4Track* track = step->GetTrack();

  // COORDINATES FOR ONLY ABSORBED UV PHOTONS
  if (volumeName == "TPB_Fiber" && track->GetParentID() == 0 &&
      track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() == "OpWLS" ) {

    G4ThreeVector position = step->GetPostStepPoint()->GetPosition();
    WritePositionToTextFile(tpbOutputFile_, position.x(), position.y());
  }

  // If you still want to check based on material for the Si detector, uncomment and use the below lines
  // G4Material* material = step->GetPreStepPoint()->GetMaterial();
  // std::string materialName = material->GetName();
  
  // Assuming that "G4_Si" is the material name and not the volume name for this condition
  if (step->GetPreStepPoint()->GetMaterial()->GetName() == "G4_Si" &&
      track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() == "OpAbsorption") {
    
    G4ThreeVector position = step->GetPreStepPoint()->GetPosition();
    WritePositionToTextFile(sipmOutputFile_, position.x(), position.y());
  }

  return true;
}


void SquareFiberSD::WritePositionToTextFile(std::ofstream& file, double x, double y) {
    if (file.is_open()) {
        file << std::fixed << std::setprecision(3)
             << x << " " << y << std::endl;
    } else {
        throw std::runtime_error("Error: Unable to write position to output file!");
    }
}



void SquareFiberSD::SetSipmPath(const G4String& path) {
  //std::cout << "SIPM_PATH=" << path << std::endl;
  sipmOutputFile_.open(path, std::ios::out | std::ios::app);
  if (!sipmOutputFile_){throw std::runtime_error("Error: Unable to open SiPM output file for writing!");}
}

void SquareFiberSD::SetTpbPath(const G4String& path) {
  //std::cout << "TPB_PATH=" << path << std::endl;
  tpbOutputFile_.open(path, std::ios::out | std::ios::app);
  if (!tpbOutputFile_){throw std::runtime_error("Error: Unable to open TPB output file for writing!");}
}


} // close namespace nexus