// // Simulating an optical fiber with its internal reflections

// #include "G4OpticalSurface.hh"
// #include "G4LogicalSkinSurface.hh"

// // Define the optical properties of the fiber material
// G4MaterialPropertiesTable* fiberMPT = new G4MaterialPropertiesTable();
// fiberMPT->AddProperty("RINDEX", photonEnergy, rIndexFiber, nEntries);
// fiberMPT->AddProperty("ABSLENGTH", photonEnergy, absorption, nEntries);
// fiberMaterial->SetMaterialPropertiesTable(fiberMPT);

// // Create the fiber geometry
// G4Tubs* fiberSolid = new G4Tubs("fiberSolid", 0, fiberRadius, fiberLength/2, 0, 2*pi);
// G4LogicalVolume* fiberLogical = new G4LogicalVolume(fiberSolid, fiberMaterial, "fiberLogical");
// G4VPhysicalVolume* fiberPhysical = new G4PVPlacement(0, G4ThreeVector(), fiberLogical, "fiberPhysical", worldLogical, false, 0);

// // Define the optical surface
// G4OpticalSurface* fiberOpticalSurface = new G4OpticalSurface("fiberOpticalSurface", glisur, polished, dielectric_dielectric, 1.0);
// fiberOpticalSurface->SetMaterialPropertiesTable(fiberMPT);
// new G4LogicalSkinSurface("fiberSurface", fiberLogical, fiberOpticalSurface);




// // Simulating reflective surfaces

// #include "G4OpticalSurface.hh"
// #include "G4LogicalBorderSurface.hh"

// // Define the optical properties of the reflective material
// G4MaterialPropertiesTable* reflectorMPT = new G4MaterialPropertiesTable();
// reflectorMPT->AddProperty("REFLECTIVITY", photonEnergy, reflectivity, nEntries);
// reflectorMaterial->SetMaterialPropertiesTable(reflectorMPT);

// // Create the reflective surface geometry
// G4Box* reflectorSolid = new G4Box("reflectorSolid", reflectorSizeX/2, reflectorSizeY/2, reflectorSizeZ/2);
// G4LogicalVolume* reflectorLogical = new G4LogicalVolume(reflectorSolid, reflectorMaterial, "reflectorLogical");
// G4VPhysicalVolume* reflectorPhysical = new G4PVPlacement(0, G4ThreeVector(reflectorPosX, reflectorPosY, reflectorPosZ), reflectorLogical, "reflectorPhysical", worldLogical, false, 0);

// // Define the optical surface
// G4OpticalSurface* reflectorOpticalSurface = new G4OpticalSurface("reflectorOpticalSurface", glisur, polished, dielectric_metal, 1.0);
// reflectorOpticalSurface->SetMaterialPropertiesTable(reflectorMPT);
// new G4LogicalBorderSurface("reflectorBorderSurface", worldPhysical, reflectorPhysical, reflectorOpticalSurface);





// // GOOD !!
// // Simulating the interface between a TPB wavelength shifter and an optical fiber
// #include "G4OpticalSurface.hh"
// #include "G4LogicalBorderSurface.hh"

// // Define the optical properties of the TPB wavelength shifter
// G4MaterialPropertiesTable* tpbMPT = new G4MaterialPropertiesTable();
// tpbMPT->AddProperty("RINDEX", photonEnergy, rIndexTPB, nEntries);
// tpbMPT->AddProperty("ABSLENGTH", photonEnergy, absorptionTPB, nEntries);
// tpbMPT->AddProperty("WLSABSLENGTH", photonEnergy, absorptionWLS, nEntries);
// tpbMPT->AddProperty("WLSCOMPONENT", photonEnergy, emissionWLS, nEntries);
// tpbMPT->AddConstProperty("WLSTIMECONSTANT", wlsTimeConstant);
// tpbMaterial->SetMaterialPropertiesTable(tpbMPT);

// // Create the TPB wavelength shifter geometry
// G4Box* tpbSolid = new G4Box("tpbSolid", tpbSizeX/2, tpbSizeY/2, tpbSizeZ/2);
// G4LogicalVolume* tpbLogical = new G4LogicalVolume(tpbSolid, tpbMaterial, "tpbLogical");
// G4VPhysicalVolume* tpbPhysical = new G4PVPlacement(0, G4ThreeVector(tpbPosX, tpbPosY, tpbPosZ), tpbLogical, "tpbPhysical", worldLogical, false, 0);

// // Define the optical surface for the interface between the TPB wavelength shifter and the optical fiber
// G4OpticalSurface* tpbFiberOpticalSurface = new G4OpticalSurface("tpbFiberOpticalSurface", glisur, polished, dielectric_dielectric, 1.0);
// tpbFiberOpticalSurface->SetMaterialPropertiesTable(tpbMPT);

// // Apply the optical surface to the interface between the TPB wavelength shifter and the optical fiber
// new G4LogicalBorderSurface("tpbFiberBorderSurface", tpbPhysical, fiberPhysical, tpbFiberOpticalSurface);





// #include "G4MaterialPropertiesTable.hh"
// #include "G4OpticalSurface.hh"
// #include "G4LogicalBorderSurface.hh"

// // Define the material properties for Xe, TPB, and PTFE
// G4MaterialPropertiesTable* xeMPT = new G4MaterialPropertiesTable();
// G4MaterialPropertiesTable* tpbMPT = new G4MaterialPropertiesTable();
// G4MaterialPropertiesTable* ptfeMPT = new G4MaterialPropertiesTable();

// // Set the refractive index, absorption length, and other relevant properties for each material
// // (Note: You should provide the actual values for each property in the corresponding arrays)
// xeMPT->AddProperty("RINDEX", photonEnergy, rIndexXe, nEntries);
// xeMPT->AddProperty("ABSLENGTH", photonEnergy, absorptionXe, nEntries);

// tpbMPT->AddProperty("RINDEX", photonEnergy, rIndexTPB, nEntries);
// tpbMPT->AddProperty("ABSLENGTH", photonEnergy, absorptionTPB, nEntries);
// tpbMPT->AddProperty("WLSABSLENGTH", photonEnergy, absorptionWLS, nEntries);
// tpbMPT->AddProperty("WLSCOMPONENT", photonEnergy, emissionWLS, nEntries);
// tpbMPT->AddConstProperty("WLSTIMECONSTANT", wlsTimeConstant);

// ptfeMPT->AddProperty("RINDEX", photonEnergy, rIndexPTFE, nEntries);
// ptfeMPT->AddProperty("REFLECTIVITY", photonEnergy, reflectivityPTFE, nEntries);

// // Assign the material properties tables to the corresponding materials
// xeMaterial->SetMaterialPropertiesTable(xeMPT);
// tpbMaterial->SetMaterialPropertiesTable(tpbMPT);
// ptfeMaterial->SetMaterialPropertiesTable(ptfeMPT);

// // Define the optical surfaces for the Xe-TPB and TPB-PTFE interfaces
// G4OpticalSurface* xeTpbOpticalSurface = new G4OpticalSurface("xeTpbOpticalSurface", glisur, polished, dielectric_dielectric, 1.0);
// xeTpbOpticalSurface->SetMaterialPropertiesTable(tpbMPT);

// G4OpticalSurface* tpbPtfeOpticalSurface = new G4OpticalSurface("tpbPtfeOpticalSurface", glisur, polished, dielectric_metal, 1.0);
// tpbPtfeOpticalSurface->SetMaterialPropertiesTable(ptfeMPT);

// // Apply the optical surfaces to the interfaces between the materials
// // (Note: You should create the volumes and their physical placements before this step)
// new G4LogicalBorderSurface("xeTpbBorderSurface", xePhysical, tpbPhysical, xeTpbOpticalSurface);
// new G4LogicalBorderSurface("tpbPtfeBorderSurface", tpbPhysical, ptfePhysical, tpbPtfeOpticalSurface);

