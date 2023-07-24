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




G4bool SquareFiberSD::ProcessHits(G4Step* step, G4TouchableHistory*) {
  G4Material* material = step->GetPreStepPoint()->GetMaterial();
  // G4Material* material = step->GetPostStepPoint()->GetMaterial();

  std::string materialName = material->GetName();
  G4Track* track = step->GetTrack();

  // std::cout << "Material: " << materialName << std::endl;
  // std::cout << "track->GetParentID() = " << track->GetParentID() << std::endl;
  // std::cout << "track->GetCurrentStepNumber() = " << track->GetCurrentStepNumber() << std::endl;


  // CASE FOR ONLY WLSE PHOTON COORDINATES
  // if (materialName == "TPB" && track->GetParentID() == 0 &&
  //    track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() == "OpWLS" ) {


  if (materialName == "TPB" && track->GetParentID() == 0) {
    G4ThreeVector position = step->GetPostStepPoint()->GetPosition();
    WritePositionToTextFile(tpbOutputFile_, position);

  }
  
  if (materialName == "G4_Si" &&
      track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() == "OpAbsorption"){

    G4ThreeVector position = step->GetPreStepPoint()->GetPosition();
    WritePositionToTextFile(sipmOutputFile_, position);

  }

  return true;
}


void SquareFiberSD::WritePositionToTextFile(std::ofstream& file, G4ThreeVector position) {
  if (file.is_open()) {
    file << position.x() << " " << position.y() << " " << position.z() << std::endl;
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