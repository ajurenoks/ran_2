import os
path_to_results_file = 'embedding_results.txt'
if not os.path.exists(path_to_results_file):
    print("Wrong results file")
    exit()

if __name__ == '__main__':
    results_file = open(path_to_results_file, "r")
    results_file_lines = results_file.readlines()
    speaker_embedding_results = {}
    for line in results_file_lines:
        conversion, dist_cos_other, dist_cos_conv, score = line.strip().split('|')
        target_speaker_conv = conversion.split('_')[-1]
        if not speaker_embedding_results.get(target_speaker_conv):
            speaker_embedding_results[target_speaker_conv] = {}
            speaker_embedding_results[target_speaker_conv]['sum_score'] = float(score)
            speaker_embedding_results[target_speaker_conv]['number'] = 1
        else:
            speaker_embedding_results[target_speaker_conv]['sum_score'] += float(score)
            speaker_embedding_results[target_speaker_conv]['number'] += 1
    results = {}
    for speaker in speaker_embedding_results:
        sum_score = speaker_embedding_results[speaker].get('sum_score')
        number = speaker_embedding_results[speaker].get('number')
        results[speaker] = sum_score/number
    sorted_results = sorted(results.items(), key=lambda x:x[1])
    print(f"Highest scores: {sorted_results[-5:]}")
    print(f"Lowest scorres: {sorted_results[:5]}")
