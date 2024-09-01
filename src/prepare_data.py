import json

def prepare_data(resume_data_path, qa_data_path, output_path):
    with open(resume_data_path, 'r') as file:
        resume_data = json.load(file)
    
    with open(qa_data_path, 'r') as file:
        qa_data = json.load(file)
    
    context = f"""
    Education:
    {resume_data['education']}

    Experience:
    """
    for exp in resume_data['experience']:
        context += f"""
        Date: {exp['date']}
        Place: {exp['place']}
        Organization: {exp['org-name']}
        Role: {exp['role']}
        Technologies: {exp['technologies']}
        Description: {exp['description']}
        """

    context += f"""
    Skills:
    Tools/Languages: {resume_data['skills']['tools_languages']}
    Technologies: {resume_data['skills']['technologies']}
    Coding Platforms: {resume_data['skills']['coding_platforms']}
    Roles and Honors: {resume_data['skills']['roles_and_honours']}
    Accomplishments: {resume_data['skills']['accomplishments']}

    Contact Information:
    Name: {resume_data['contact_information']['Name']}
    Email: {resume_data['contact_information']['Email']}
    Phone Number: {resume_data['contact_information']['Phone Number']}
    Portfolio/LinkedIn: {resume_data['contact_information']['Portfolio_LinkedIn']}
    """

    training_data = []
    for qa in qa_data:
        question = qa['question']
        answer = qa['answer']
        training_data.append({
            'context': context,
            'question': question,
            'answer': answer
        })
    
    with open(output_path, 'w') as file:
        json.dump(training_data, file, indent=4)

if __name__ == "__main__":
    prepare_data('data/resume_data.json', 'data/custom_qa.json', 'data/fine_tuning_data.json')
