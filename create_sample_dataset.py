#!/usr/bin/env python3
"""
Create a sample fake job postings dataset for testing Phase 1
This creates a dataset with the same structure as the Kaggle dataset
"""

import pandas as pd
import numpy as np

def create_sample_dataset():
    """
    Create a sample dataset with fake job postings
    """
    np.random.seed(42)  # For reproducibility
    
    # Sample data - mix of real and fake job postings
    sample_data = [
        {
            'job_id': 1,
            'title': 'Marketing Intern',
            'location': 'US, NY, New York',
            'department': 'Marketing',
            'salary_range': '',
            'company_profile': "We're Food52, and we've created a groundbreaking and award-winning cooking site. We support, connect, and celebrate home cooks, and give them everything they need in one place.",
            'description': 'Food52, a fast-growing, James Beard Award-winning online food community is currently interviewing full- and part-time unpaid interns to work in a small team of editors, executives, and developers in its New York City headquarters.',
            'requirements': 'Experience with content management systems a major plus. Familiar with the Food52 editorial voice and aesthetic. Loves food, appreciates the importance of home cooking.',
            'benefits': '',
            'telecommuting': 0,
            'has_company_logo': 1,
            'has_questions': 0,
            'employment_type': 'Other',
            'required_experience': 'Internship',
            'required_education': '',
            'industry': '',
            'function': 'Marketing',
            'fraudulent': 0
        },
        {
            'job_id': 2,
            'title': 'Customer Service - Cloud Video Production',
            'location': 'NZ, Auckland',
            'department': 'Success',
            'salary_range': '$40,000-$55,000',
            'company_profile': '90 Seconds, the worlds Cloud Video Production Service. 90 Seconds makes video production fast, affordable, and all managed seamlessly in the cloud.',
            'description': 'Our rapidly expanding business is looking for a talented Project Manager to manage the successful delivery of video projects, manage client communications and drive the production process.',
            'requirements': 'Client focused - excellent customer service and communication skills. Outstanding computer knowledge and experience using online software.',
            'benefits': 'Experience working on projects located around the world with an international brand. Opportunity to drive and grow production function.',
            'telecommuting': 0,
            'has_company_logo': 1,
            'has_questions': 0,
            'employment_type': 'Full-time',
            'required_experience': 'Not Applicable',
            'required_education': '',
            'industry': 'Marketing and Advertising',
            'function': 'Customer Service',
            'fraudulent': 0
        },
        {
            'job_id': 3,
            'title': 'MAKE $5000 WEEKLY FROM HOME!!!',
            'location': 'Anywhere',
            'department': '',
            'salary_range': '$5000/week',
            'company_profile': 'Work from home opportunity! No experience needed! Guaranteed income!',
            'description': 'URGENT: We need people to work from home immediately! Make $5000 per week guaranteed! No skills required! Just copy and paste! Click link now!',
            'requirements': 'No requirements! Anyone can do this! Just need computer and internet!',
            'benefits': 'GUARANTEED $5000 per week! Work from anywhere! Be your own boss! Financial freedom!',
            'telecommuting': 1,
            'has_company_logo': 0,
            'has_questions': 0,
            'employment_type': 'Full-time',
            'required_experience': 'Entry Level',
            'required_education': '',
            'industry': '',
            'function': '',
            'fraudulent': 1
        },
        {
            'job_id': 4,
            'title': 'Software Engineer',
            'location': 'CA, San Francisco',
            'department': 'Engineering',
            'salary_range': '$120,000-$160,000',
            'company_profile': 'We are a leading technology company focused on building innovative solutions that help businesses scale and grow.',
            'description': 'We are seeking a talented Software Engineer to join our engineering team. You will work on developing scalable web applications using modern technologies.',
            'requirements': 'Bachelor\'s degree in Computer Science or related field. 3+ years of experience with Python, JavaScript, and React. Strong problem-solving skills.',
            'benefits': 'Competitive salary, health insurance, dental, vision, 401k matching, flexible PTO, remote work options.',
            'telecommuting': 1,
            'has_company_logo': 1,
            'has_questions': 1,
            'employment_type': 'Full-time',
            'required_experience': 'Mid Level',
            'required_education': 'Bachelor\'s Degree',
            'industry': 'Technology',
            'function': 'Engineering',
            'fraudulent': 0
        },
        {
            'job_id': 5,
            'title': 'Data Entry Specialist URGENT HIRING',
            'location': 'Remote Worldwide',
            'department': '',
            'salary_range': '$3000-$8000',
            'company_profile': 'International company needs data entry specialists immediately. Very high pay for simple work.',
            'description': 'URGENT HIRING: Need data entry specialists NOW! Simple copy/paste work! Very high pay! No experience needed! Must start immediately!',
            'requirements': 'Must have computer. Must start today. No other requirements.',
            'benefits': 'Very high pay! Work from home! Flexible hours! Easy work!',
            'telecommuting': 1,
            'has_company_logo': 0,
            'has_questions': 0,
            'employment_type': 'Full-time',
            'required_experience': 'Entry Level',
            'required_education': '',
            'industry': '',
            'function': 'Administrative',
            'fraudulent': 1
        },
        {
            'job_id': 6,
            'title': 'Product Manager',
            'location': 'WA, Seattle',
            'department': 'Product',
            'salary_range': '$130,000-$170,000',
            'company_profile': 'Amazon is a Fortune 500 company and leading e-commerce platform. We are customer obsessed and strive to be Earth\'s Most Customer-Centric Company.',
            'description': 'We are looking for a Product Manager to lead product strategy and development for our consumer marketplace. You will work with cross-functional teams to deliver innovative features.',
            'requirements': 'MBA or equivalent experience. 5+ years of product management experience. Strong analytical skills and customer obsession. Experience with agile development.',
            'benefits': 'Competitive compensation, comprehensive health benefits, retirement savings plan, paid time off, employee stock purchase plan.',
            'telecommuting': 0,
            'has_company_logo': 1,
            'has_questions': 1,
            'employment_type': 'Full-time',
            'required_experience': 'Senior Level',
            'required_education': 'Master\'s Degree',
            'industry': 'Technology',
            'function': 'Product Management',
            'fraudulent': 0
        },
        {
            'job_id': 7,
            'title': 'EARN $500 DAILY WITH CRYPTO!!!',
            'location': 'Online',
            'department': '',
            'salary_range': '$500/day',
            'company_profile': 'Revolutionary crypto investment opportunity! Join thousands making massive profits daily!',
            'description': 'ATTENTION: Crypto millionaires are sharing their secret! Earn $500+ daily with automated crypto trading! No experience needed! Join now before spots fill up!',
            'requirements': 'Just $100 to start! Anyone can do this! Works on autopilot!',
            'benefits': 'Guaranteed daily profits! Financial freedom! Retire early! Join crypto millionaires!',
            'telecommuting': 1,
            'has_company_logo': 0,
            'has_questions': 0,
            'employment_type': 'Other',
            'required_experience': 'Entry Level',
            'required_education': '',
            'industry': 'Finance',
            'function': '',
            'fraudulent': 1
        },
        {
            'job_id': 8,
            'title': 'UX Designer',
            'location': 'CA, Palo Alto',
            'department': 'Design',
            'salary_range': '$100,000-$140,000',
            'company_profile': 'Google is a multinational technology company that specializes in Internet-related services and products.',
            'description': 'We are seeking a UX Designer to join our design team. You will be responsible for creating intuitive and engaging user experiences for our products.',
            'requirements': 'Bachelor\'s degree in Design, HCI, or related field. 3+ years of UX design experience. Proficiency in design tools like Figma, Sketch. Strong portfolio.',
            'benefits': 'Competitive salary, comprehensive health coverage, retirement plan, free meals, on-site fitness facilities, learning and development opportunities.',
            'telecommuting': 0,
            'has_company_logo': 1,
            'has_questions': 1,
            'employment_type': 'Full-time',
            'required_experience': 'Mid Level',
            'required_education': 'Bachelor\'s Degree',
            'industry': 'Technology',
            'function': 'Design',
            'fraudulent': 0
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Add some additional rows by duplicating and modifying existing ones
    additional_rows = []
    for i in range(50):  # Create 50 more rows
        # Randomly select a base row
        base_idx = np.random.randint(0, len(sample_data))
        new_row = sample_data[base_idx].copy()
        new_row['job_id'] = len(sample_data) + i + 1
        
        # Add some variation to the title
        if new_row['fraudulent'] == 1:
            new_row['title'] = new_row['title'] + f" - Opportunity {i+1}"
        else:
            new_row['title'] = new_row['title'] + f" - Position {i+1}"
        
        additional_rows.append(new_row)
    
    # Add the additional rows
    additional_df = pd.DataFrame(additional_rows)
    df = pd.concat([df, additional_df], ignore_index=True)
    
    print(f"Created sample dataset with {len(df)} rows")
    print(f"Fraudulent distribution:")
    print(df['fraudulent'].value_counts())
    
    return df

def main():
    """
    Create and save the sample dataset
    """
    print("Creating sample fake job postings dataset...")
    
    # Create sample dataset
    df = create_sample_dataset()
    
    # Save to CSV
    output_file = 'data/fake_job_postings.csv'
    df.to_csv(output_file, index=False)
    print(f"Sample dataset saved to: {output_file}")
    
    # Print first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    print(f"\nColumns: {list(df.columns)}")

if __name__ == "__main__":
    main()