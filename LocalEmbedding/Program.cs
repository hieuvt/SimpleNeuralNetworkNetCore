// Đường dẫn tới mô hình FastText
using FastText.NetWrapper;

//https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.bin.gz
string modelPath = @"C:\dev\Fasttext\cc.vi.300.bin";
using (var fastText = new FastTextWrapper())
{
    fastText.LoadModel(modelPath);

    // Từ cần tìm đồng nghĩa
    string word = "hạnh_phúc";
    int topK = 5; // Số từ đồng nghĩa muốn tìm

    // Lấy vector embedding của từ cần tìm
    var wordVector = fastText.GetWordVector(word);
    //các từ gần nghĩa nhất
    var closeWords = fastText.GetNearestNeighbours(word, 5);

    foreach (var closeWord in closeWords)
    {
        Console.WriteLine(closeWord.Label + "-" + closeWord.Probability);
    }

    // Danh sách các từ cần kiểm tra (có thể là từ điển từ tiếng Việt, hoặc một số từ phổ biến)
    List<string> vocabulary = new List<string> { "vui", "buồn", "sung_sướng", "hạnh_phúc", "đau_khổ", "vui_vẻ", "an_vui" };

    // Tính toán độ tương đồng cosine và tìm từ đồng nghĩa
    var synonyms = vocabulary
        .Where(vocabWord => vocabWord != word) // Loại trừ từ gốc
        .Select(vocabWord => new
        {
            Word = vocabWord,
            Similarity = CosineSimilarity(wordVector, fastText.GetWordVector(vocabWord))
        })
        .OrderByDescending(x => x.Similarity)
        .Take(topK);

    // In ra kết quả
    Console.WriteLine($"Từ đồng nghĩa gần nhất với '{word}':");
    foreach (var synonym in synonyms)
    {
        Console.WriteLine($"{synonym.Word} (Độ tương đồng: {synonym.Similarity:F2})");
    }
}

// Hàm tính độ tương đồng cosine giữa hai vector
static float CosineSimilarity(float[] vectorA, float[] vectorB)
{
    float dotProduct = 0;
    float magnitudeA = 0;
    float magnitudeB = 0;

    for (int i = 0; i < vectorA.Length; i++)
    {
        dotProduct += vectorA[i] * vectorB[i];
        magnitudeA += vectorA[i] * vectorA[i];
        magnitudeB += vectorB[i] * vectorB[i];
    }

    return dotProduct / ((float)Math.Sqrt(magnitudeA) * (float)Math.Sqrt(magnitudeB));
}

