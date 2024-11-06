#include "TH1.h"
#include "TFile.h"
#include "TCanvas.h"
#include <string>
#include <TMath.h>

using namespace std;

void rebin(TH1D * h_temp, TH1D* h){

  for (int i = 0; i < h->GetXaxis()->GetNbins(); i++){
    if(i < h_temp->GetXaxis()->GetNbins()) {h->SetBinContent(i+1, h_temp->GetBinContent(i+1));}
    else{h->SetBinContent(i+1, 0);}
  }

}

void make_canvas (string rootfile1, string rootfile2, string hist1, string hist2, string legend1, string legend2, bool LiHe){
  int nbins = 410;
  float ene_min = 0.8;
  float ene_max = 9.0;
  if(LiHe){
    nbins = 520;
    ene_max = 12.0;
  }

  TFile *f1 = new TFile(rootfile1.c_str(), "read");
  TH1D *h1_temp = (TH1D*)f1->Get(hist1.c_str());

  TFile *f2 = new TFile(rootfile2.c_str(), "read");
  TH1D *h2_temp = (TH1D*)f2->Get(hist2.c_str());


  TH1D* h1 = new TH1D("h1", "h1", nbins, ene_min, ene_max);
  TH1D* h2 = new TH1D("h2", "h2", nbins, ene_min, ene_max);

 //if(h1_temp->GetXaxis()->GetNbins() != nbins){
   rebin(h1_temp, h1);
 //}
 //else{
//   h1 = h1_temp;
 //}
 //if(h2_temp->GetXaxis()->GetNbins() != nbins){
   rebin(h2_temp, h2);
 //}
 //else{
  // h2 = h2_temp;
 //}
//  std::cout << h1->GetBinCenter(1) << " " << h1->GetBinCenter(nbins) << std::endl;
//  std::cout << h2->GetBinCenter(1) << " " << h2->GetBinCenter(nbins) << std::endl;
  // if(h1_temp->GetXaxis()->GetNbins() != h2_temp->GetXaxis()->GetNbins()){
  //   int max_bins = max(h1_temp->GetXaxis()->GetNbins(), h2_temp->GetXaxis()->GetNbins());
  //   if (h1_temp->GetXaxis()->GetNbins() != max_bins){
  //     h1 = new TH1D("h1", "h1", max_bins, h2_temp->GetBinCenter(1)-0.01, h2_temp->GetBinCenter(max_bins)+0.01);
  //     for (int i =0; i < max_bins; i++){
  //       if (i < h1_temp->GetXaxis()->GetNbins()){ h1->SetBinContent(i+1, h1_temp->GetBinContent(h1->FindBin(h1->GetBinCenter(i+1))));}
  //       else{h1->SetBinContent(i+1, 0);}
  //     }
  //     h2 = h2_temp;
  //   }
  //   if (h2_temp->GetXaxis()->GetNbins() != max_bins){
  //     h2 = new TH1D("h2", "h2", max_bins, h1_temp->GetBinCenter(1)-0.01, h1_temp->GetBinCenter(max_bins)+0.01);
  //     for (int i =0; i < max_bins; i++){
  //       if (i < h2_temp->GetXaxis()->GetNbins()){ h2->SetBinContent(i+1, h2_temp->GetBinContent(h2->FindBin(h2->GetBinCenter(i+1))));}
  //       else{h2->SetBinContent(i+1, 0);}
  //     }
  //     h1 = h1_temp;
  //   }
  //   else{
  //     h1 = h1_temp;
  //     h2 = h2_temp;
  //   }
  // }


  h1->Scale(1./h1->Integral());
  h1->GetYaxis()->SetTitle("a.u.");
  h1->GetXaxis()->SetTitle("");
  h1->GetYaxis()->SetTitleSize(0.06);
  h1->GetYaxis()->SetLabelSize(0.05);
  h1->GetXaxis()->SetLabelSize(0.05);
  h1->GetYaxis()->SetTitleOffset(0.8);
  h1->SetTitle("");
  h2->Scale(1./h2->Integral());

  /*for (int i = 0; i < h1->GetXaxis()->GetNbins(); i++){
    std::cout << h2->GetBinContent(i+1)/h1->GetBinContent(i+1) << ",\n";
  }
  */
  TCanvas *c = new TCanvas("", "", 1500, 1100);
  gStyle->SetOptStat(0);
  c->Divide(1,2);
  TPad* pad1 = (TPad*)c->cd(1);
  TPad* pad2 = (TPad*)c->cd(2);
  c->cd(); pad1->Draw(); pad2->Draw();
  pad1->cd();
  h1->SetLineColor(4); h1->SetLineWidth(2); h2->Draw("hist");
  h2->SetLineColor(2); h2->SetLineWidth(2); h1->Draw("hist && same");
  TLegend *legend = new TLegend(0.4, 0.5, 0.9, 0.8);
  TLegendEntry *entry1 =legend->AddEntry(h1, legend1.c_str(), "");
  TLegendEntry *entry2 =legend->AddEntry(h2, legend2.c_str(), "");

  entry1->SetTextColor(4);
  entry2->SetTextColor(2);
  entry1->SetTextSize(0.06);
  entry2->SetTextSize(0.06);
  legend->SetBorderSize(0);


  legend->Draw();
  TH1F *hres = new TH1F("hres", "hres",nbins, ene_min, ene_max);
  hres->Sumw2();
  pad2->cd();
  hres->GetYaxis()->SetTitle("Rel difference [%]");
  hres->GetXaxis()->SetTitle("Reconstructed Energy (MeV)");
  hres->GetYaxis()->SetTitleSize(0.06);
  hres->GetYaxis()->SetTitleOffset(0.8);
  hres->GetXaxis()->SetTitleSize(0.06);
  hres->GetXaxis()->SetTitleOffset(0.8);
  hres->GetYaxis()->SetLabelSize(0.05);
  hres->GetXaxis()->SetLabelSize(0.05);
  hres->SetTitle("");
  hres->Add(h1, h2, 1, -1); hres->Divide(h1);
  std::cout << h1->GetXaxis()->GetNbins() << " " << h2->GetXaxis()->GetNbins() << std::endl;
  hres->SetLineColor(1); hres->SetLineWidth(2);hres->Draw("hist");
  hres->Scale(100.);
  hres->GetYaxis()->SetRangeUser(-10, 10);
  TLine *line = new TLine(hres->GetXaxis()->GetXmin(), 0.0, hres->GetXaxis()->GetXmax(), 0.0);


  line->SetLineColor(kGreen);
  line->SetLineWidth(2);
  line->SetLineStyle(2);
  line->Draw();


}

void comp_hist_general(){

   //make_canvas("Sindhu_spectra_Oct28.root", "/home/sindhu/Downloads/j22_reac_spec_wcorr.root", "rosc", "hreac_wcorr", "Sindhu old", "Yury", false);
  // make_canvas("Sindhu_spectra_Oct28.root", "/home/sindhu/Downloads/j22_reac_spec_wcorr.root", "rosc_Enfixed_new2", "hreac_wcorr", "Sindhu Tn fixed", "Yury", false);
   //make_canvas("Sindhu_spectra_Oct28.root", "/home/sindhu/Downloads/j22_reac_spec_wcorr.root", "rosc_newpos_DYB", "hreac_wcorr", "Sindhu 2D hist", "Yury", false);
  //make_canvas("Sindhu_spectra_Oct28.root", "Sindhu_spectra_Oct28.root", "rosc","rosc_Enfixed_new2", "old analytical", "with Tn fixed", false);
  //make_canvas("Sindhu_spectra_Oct28.root", "Sindhu_spectra_Oct28.root", "rosc", "rosc_newpos_DYB", "old analytical", "Epos from 2D hist", false);
   make_canvas("Sindhu_spectra_Oct28.root", "Sindhu_spectra_Oct28.root", "rosc_newpos", "rosc_newpos_DYB", "Strumia Vissani", "Vogel Beacom", false);

}
