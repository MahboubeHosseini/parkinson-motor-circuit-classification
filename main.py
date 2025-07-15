"""
Main execution script for Parkinson's Disease Subtype Classification
Run this script to execute the complete analysis pipeline.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import *
from analysis import PaperAnalysis
from visualizations import (create_summary_metrics_table, create_detailed_classifier_tables,
                           create_roc_curves, create_pr_curves, create_calibration_curves,
                           create_dca_curves, create_overfitting_heatmap, 
                           create_radar_charts, create_combined_radar_chart,
                           create_motor_circuit_top_features_analysis,
                           create_motor_circuit_feature_family_analysis,
                           create_individual_classifier_feature_importance_plots,
                           create_comprehensive_consistency_analysis)

# Try to import permutation importance
try:
    from sklearn.inspection import permutation_importance
    PERMUTATION_AVAILABLE = True
    print("✅ Sklearn permutation importance available")
except ImportError:
    PERMUTATION_AVAILABLE = False
    print("⚠️  Using enhanced manual permutation importance implementation")


def run_paper_analysis():
    """Main function to run enhanced paper analysis"""
   
    print("=" * 80)
    print("🧠 Parkinson's Disease Subtype Classification Analysis")
    print("=" * 80)
   
    # Create analyzer
    analyzer = PaperAnalysis(
        metadata_paths=METADATA_PATHS,
        base_paths=BASE_PATHS,
        roi_list=MOTOR_ROI
    )
   
    # Run complete enhanced analysis
    all_results = analyzer.run_complete_analysis()
    
    if all_results is None:
        print("\n❌ Analysis failed. Check error messages above.")
        return None, []
    
    print("\n📊 Creating visualizations and tables...")
    
    generated_files = []
    
    try:
        # 1. Summary metrics table
        print("Creating summary metrics table...")
        summary_df, summary_file = create_summary_metrics_table(all_results, analyzer.classifiers)
        generated_files.append(summary_file)
        
        # 2. Detailed classifier tables
        print("Creating detailed classifier tables...")
        detailed_file = create_detailed_classifier_tables(all_results, analyzer.classifiers)
        generated_files.append(detailed_file)
        
        # 3. ROC curves
        print("Creating ROC curves...")
        roc_file = create_roc_curves(all_results, analyzer.scenarios, analyzer.labels)
        generated_files.append(roc_file)
        
        # 4. Precision-Recall curves
        print("Creating Precision-Recall curves...")
        pr_file = create_pr_curves(all_results, analyzer.labels)
        generated_files.append(pr_file)
        
        # 5. Calibration curves
        print("Creating calibration curves...")
        cal_file = create_calibration_curves(all_results)
        generated_files.append(cal_file)
        
        # 6. Decision Curve Analysis
        print("Creating DCA curves...")
        dca_file = create_dca_curves(all_results, analyzer.labels)
        generated_files.append(dca_file)
        
        # 7. Overfitting heatmap
        print("Creating overfitting heatmap...")
        overfitting_file = create_overfitting_heatmap(all_results, analyzer.classifiers, analyzer.scenarios)
        generated_files.append(overfitting_file)
        
        # 8. Performance radar charts
        print("Creating performance radar charts...")
        radar_file = create_radar_charts(all_results, analyzer.classifiers)
        generated_files.append(radar_file)
        
        # 9. Combined radar chart (best scenarios)
        print("Creating combined radar chart...")
        combined_radar_file = create_combined_radar_chart(all_results, analyzer.classifiers)
        generated_files.append(combined_radar_file)
        
        # 10. Top Motor-Circuit Features Analysis
        print("Creating Top Motor-Circuit features analysis...")
        top_fig, top_ax = create_motor_circuit_top_features_analysis(
            all_results, analyzer.classifiers, analyzer.scenarios, 
            save_path='motor_circuit_top_features_analysis.png'
        )
        generated_files.append('motor_circuit_top_features_analysis.png')
        
        # 11. Motor Circuit Feature Family Analysis
        print("Creating motor circuit feature family analysis...")
        family_analysis = create_motor_circuit_feature_family_analysis(
            all_results, analyzer.classifiers, analyzer.scenarios
        )
        if family_analysis:
            generated_files.append(family_analysis['plot_file'])
            generated_files.append(family_analysis['summary_file'])
        
        # 12. Individual classifier feature importance plots
        print("Creating individual classifier feature importance plots...")
        individual_files = create_individual_classifier_feature_importance_plots(
            all_results, analyzer.classifiers, analyzer.scenarios
        )
        generated_files.extend(individual_files)
        
        # 13. Comprehensive consistency analysis (chord diagrams)
        print("Creating comprehensive consistency analysis...")
        chord_files = create_comprehensive_consistency_analysis(all_results)
        generated_files.extend(chord_files)
        
        # Calculate overall statistics
        total_motor_circuit_wins = 0
        total_comparisons = 0
        
        for scenario_name, results in all_results.items():
            for clf in analyzer.classifiers.keys():
                single_roi_aucs = results['single_roi'].get(f'{clf}_auc', [])
                motor_circuit_aucs = results['motor_circuit'].get(f'{clf}_auc', [])
                if single_roi_aucs and motor_circuit_aucs:
                    total_comparisons += 1
                    if np.mean(motor_circuit_aucs) > np.mean(single_roi_aucs):
                        total_motor_circuit_wins += 1
        
        overall_win_rate = total_motor_circuit_wins / total_comparisons * 100 if total_comparisons > 0 else 0
        
        print(f"\n🎉 SUCCESS! Enhanced analysis completed.")
        print(f"📁 Generated {len(generated_files)} files")
        print(f"📊 Sample Size: {analyzer.sample_size}")
        print(f"🔬 Scenarios Tested: {len(analyzer.scenarios)}")
        print(f"🤖 Classifiers Tested: {len(analyzer.classifiers)}")
        print(f"🏆 Motor-Circuit Win Rate: {overall_win_rate:.1f}% ({total_motor_circuit_wins}/{total_comparisons})")
        
        print(f"\n📄 Generated Files:")
        for file in generated_files:
            print(f"  • {file}")
            
    except Exception as e:
        print(f"⚠️  Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
   
    return all_results, generated_files


if __name__ == "__main__":
    # Run the complete analysis
    results, files = run_paper_analysis()
    
    if results:
        print(f"\n✨ Analysis completed successfully!")
        print(f"📁 Files saved: {files}")
        print(f"\n🔬 To run again, simply execute: python main.py")
        print(f"📚 Check README.md for detailed documentation")
    else:
        print(f"\n💥 Analysis failed!")
        print(f"🔧 Check your data paths in config.py")
        print(f"📋 Ensure all dependencies are installed: pip install -r requirements.txt")