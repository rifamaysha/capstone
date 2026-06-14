import { useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  FileImage,
  RefreshCw,
  ScanLine,
  Smartphone,
  UploadCloud,
  ArrowRight,
  ClipboardPenLine,
} from "lucide-react";
import Header from "../components/Header.jsx";
import { extractTransaction, saveTransaction } from "../api/client.js";
import { categories } from "../styles/theme.js";

const DOC_TYPES = [
  { value: "auto", label: "Otomatis", icon: RefreshCw },
  { value: "receipt", label: "Struk Belanja", icon: FileImage },
  { value: "screenshot", label: "Screenshot Pembayaran", icon: Smartphone },
];

const INPUT_METHODS = [
  { value: "scan", label: "Scan Transaksi", icon: ScanLine },
  { value: "manual", label: "Input Manual", icon: ClipboardPenLine },
];

const EMPTY_FORM = {
  merchant: "",
  amount: "",
  date: "",
  category: "lainnya",
  notes: "",
};

function formatRp(v) {
  return `Rp ${Number(v || 0).toLocaleString("id-ID")}`;
}

function statusLabel(status, form) {
  if (status === "failed") return "Belum terbaca";
  if (status === "needs_review") return "Perlu dicek";
  if (!form.merchant || !form.amount) return "Perlu dicek";
  return "Terbaca";
}

function DetectionBadge({ result, selectedType }) {
  const type = result?.document_type || "unknown";
  const label = result?.document_type_label || "Perlu dicek manual";
  const isManualType = selectedType !== "auto" || result?.document_type_source === "manual";

  if (type === "unknown") {
    return (
      <span className="detect-badge detect-badge-unknown">
        <AlertTriangle size={15} />
        Jenis transaksi perlu dicek
      </span>
    );
  }

  const Icon = type === "receipt" ? FileImage : Smartphone;
  return (
    <span className={`detect-badge detect-badge-${type}`}>
      <Icon size={15} />
      {isManualType ? `Diproses sebagai ${label}` : `Terdeteksi sebagai ${label}`}
    </span>
  );
}

export default function UploadProcess({ onNavigate }) {
  const inputRef = useRef(null);
  const [inputMethod, setInputMethod] = useState("scan");
  const [docType, setDocType] = useState("auto");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [isDrag, setIsDrag] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [result, setResult] = useState(null);
  const [message, setMessage] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [batchDrafts, setBatchDrafts] = useState([]);
  const [batchProgress, setBatchProgress] = useState("");
  const [queuedFiles, setQueuedFiles] = useState([]);
  const [activeDraftIndex, setActiveDraftIndex] = useState(0);
  const [filePickMode, setFilePickMode] = useState("single");
  const [batchSaveSummary, setBatchSaveSummary] = useState(null);
  const [manualForm, setManualForm] = useState(EMPTY_FORM);
  const [manualMessage, setManualMessage] = useState("");
  const [manualError, setManualError] = useState("");
  const [manualSaved, setManualSaved] = useState(false);
  const [manualSavedData, setManualSavedData] = useState(null);

  const resultToForm = (res) => ({
    merchant: res.merchant || "",
    amount: res.amount > 0 ? String(Math.round(res.amount)) : "",
    date: res.date || "",
    category: res.category || "lainnya",
    notes: "",
  });

  const isBatchReview = batchDrafts.length > 1;

  const openFilePicker = (mode = "single") => {
    setFilePickMode(mode);
    inputRef.current?.click();
  };

  const applyDraft = (draft, index) => {
    if (!draft) return;
    setActiveDraftIndex(index);
    setFile(draft.fileObject || null);
    setPreview(draft.preview || "");
    setResult(draft.result || null);
    setForm(draft.form || EMPTY_FORM);
    setSaveError(draft.saveError || "");
  };

  const updateActiveDraftForm = (nextForm) => {
    setForm(nextForm);
    if (!isBatchReview) return;
    setBatchDrafts((drafts) => drafts.map((draft, index) => (
      index === activeDraftIndex ? { ...draft, form: nextForm, saved: false, saveError: "" } : draft
    )));
  };

  const addFilesToQueue = (files) => {
    const selectedFiles = Array.from(files || []).filter((item) => item?.type?.startsWith("image/"));
    if (!selectedFiles.length) return;
    setBatchSaveSummary(null);
    const validFiles = selectedFiles.filter((item) => item.size <= 10 * 1024 * 1024);
    if (validFiles.length !== selectedFiles.length) {
      setMessage("Ukuran file terlalu besar. Pilih gambar sampai 10 MB.");
    }
    setQueuedFiles((current) => {
      const next = [...current, ...validFiles].slice(0, 3);
      if (current.length + validFiles.length > 3) {
        setMessage("Maksimal 3 transaksi per proses.");
      }
      return next;
    });
  };

  const clearQueue = () => {
    setQueuedFiles([]);
    setBatchSaveSummary(null);
    setMessage("");
  };

  const startExtract = async (targetFile = file, options = {}) => {
    const draftMode = isBatchReview && !options.single;
    const draft = draftMode ? batchDrafts[activeDraftIndex] : null;
    const fileToExtract = targetFile || draft?.fileObject;
    if (!fileToExtract) return;
    setExtracting(true);
    if (!options.keepResult) setResult(null);
    if (!options.keepMessage) setMessage("");
    setSaveError("");

    try {
      const res = await extractTransaction(fileToExtract, docType);
      setResult(res);
      const failed = !res.success || res.status === "failed";
      const nextForm = resultToForm(res);
      setForm(nextForm);
      if (draftMode) {
        setBatchDrafts((drafts) => drafts.map((item, index) => (
          index === activeDraftIndex
            ? { ...item, result: res, form: nextForm, saved: false, saveError: "" }
            : item
        )));
      }
      const hasPartial = Boolean(nextForm.merchant || nextForm.amount || nextForm.date);
      if (failed && !hasPartial) {
        setMessage("Transaksi belum berhasil dibaca. Isi detail transaksi secara manual.");
      } else if (failed || res.status === "needs_review") {
        setMessage("Beberapa data belum terbaca. Lengkapi sebelum menyimpan.");
      } else if (!nextForm.merchant || !nextForm.amount || !nextForm.date) {
        setMessage("Beberapa data belum terbaca. Lengkapi sebelum menyimpan.");
      }
    } catch (error) {
      const errorMessage = error?.message || "Transaksi belum berhasil dibaca. Isi detail transaksi secara manual.";
      const isTimeout = errorMessage.toLowerCase().includes("terlalu lama");
      setResult({
        success: false,
        status: isTimeout ? "needs_review" : "failed",
        document_type: "unknown",
        document_type_label: "Perlu dicek manual",
        warnings: [errorMessage],
      });
      setForm(EMPTY_FORM);
      if (draftMode) {
        setBatchDrafts((drafts) => drafts.map((item, index) => (
          index === activeDraftIndex
            ? {
                ...item,
                result: {
                  success: false,
                  status: isTimeout ? "needs_review" : "failed",
                  document_type: "unknown",
                  document_type_label: "Perlu dicek manual",
                  warnings: [errorMessage],
                },
                form: EMPTY_FORM,
                saved: false,
                saveError: isTimeout ? "Proses terlalu lama. Isi manual atau coba gambar lain." : "Transaksi belum berhasil dibaca.",
              }
            : item
        )));
      }
      setMessage(isTimeout ? errorMessage : "Transaksi belum berhasil dibaca. Isi detail transaksi secara manual.");
    } finally {
      setExtracting(false);
    }
  };

  const processFilesSequential = async (files) => {
    const selectedFiles = Array.from(files || []).filter((item) => item?.type?.startsWith("image/"));
    if (!selectedFiles.length) return;
    const queue = selectedFiles.slice(0, 3);
    const limited = selectedFiles.length > 3;
    const firstOversized = queue.find((item) => item.size > 10 * 1024 * 1024);
    if (firstOversized) {
      setMessage("Ukuran file terlalu besar. Pilih gambar sampai 10 MB.");
      return;
    }

    if (queue.length === 1) {
      setQueuedFiles([]);
      handleFile(queue[0]);
      return;
    }

    setExtracting(true);
    setSaved(false);
    setResult(null);
    setSaveError("");
    setForm(EMPTY_FORM);
    setBatchDrafts([]);
    setQueuedFiles([]);
    setActiveDraftIndex(0);
    setBatchSaveSummary(null);
    setMessage(limited ? "Maksimal 3 transaksi per proses." : "");

    const drafts = [];
    for (const [index, nextFile] of queue.entries()) {
      const nextPreview = URL.createObjectURL(nextFile);
      setFile(nextFile);
      setPreview(nextPreview);
      setActiveDraftIndex(index);
      setBatchProgress(`Memproses ${index + 1} dari ${queue.length} transaksi...`);
      try {
        const res = await extractTransaction(nextFile, docType);
        const nextForm = resultToForm(res);
        drafts.push({
          file: nextFile.name,
          fileObject: nextFile,
          preview: nextPreview,
          result: res,
          form: nextForm,
          saved: false,
          saveError: "",
        });
        setBatchDrafts([...drafts]);
        setResult(res);
        setForm(nextForm);
      } catch (error) {
        const errorMessage = error?.message || "Transaksi belum berhasil dibaca.";
        const isTimeout = errorMessage.toLowerCase().includes("terlalu lama");
        const failedResult = {
          success: false,
          status: isTimeout ? "needs_review" : "failed",
          document_type: "unknown",
          document_type_label: "Perlu dicek manual",
          warnings: [errorMessage],
        };
        const failedForm = { ...EMPTY_FORM };
        drafts.push({
          file: nextFile.name,
          fileObject: nextFile,
          preview: nextPreview,
          result: failedResult,
          form: failedForm,
          saved: false,
          saveError: isTimeout ? "Proses terlalu lama. Isi manual atau coba gambar lain." : "Transaksi belum berhasil dibaca.",
        });
        setBatchDrafts([...drafts]);
        setResult(failedResult);
        setForm(failedForm);
      }
    }

    setBatchProgress("");
    setExtracting(false);
    if (drafts.length) {
      setActiveDraftIndex(0);
      setFile(drafts[0].fileObject);
      setPreview(drafts[0].preview);
      setResult(drafts[0].result);
      setForm(drafts[0].form);
    }
    const needsReview = drafts.some(({ result: draftResult, form: draftForm }) => (
      draftResult?.status !== "extracted" || !draftForm.merchant || !draftForm.amount || !draftForm.date
    ));
    const timeoutCount = drafts.filter((draft) => (
      (draft.saveError || "").toLowerCase().includes("terlalu lama")
      || (draft.result?.warnings || []).some((warning) => String(warning).toLowerCase().includes("terlalu lama"))
    )).length;
    if (limited) {
      setMessage("Maksimal 3 transaksi per proses.");
    } else if (timeoutCount > 0) {
      setMessage(`${timeoutCount} transaksi belum berhasil dibaca karena proses terlalu lama.`);
    } else if (needsReview) {
      setMessage("Beberapa data belum terbaca. Lengkapi sebelum menyimpan.");
    } else {
      setMessage("");
    }
  };

  const handleFile = (nextFile) => {
    if (!nextFile || !nextFile.type.startsWith("image/")) return;
    if (nextFile.size > 10 * 1024 * 1024) {
      setMessage("Ukuran file terlalu besar. Pilih gambar sampai 10 MB.");
      return;
    }

    const nextPreview = URL.createObjectURL(nextFile);
    setFile(nextFile);
    setPreview(nextPreview);
    setSaved(false);
    setResult(null);
    setMessage("");
    setSaveError("");
    setForm(EMPTY_FORM);
    setBatchDrafts([]);
    setBatchProgress("");
    setQueuedFiles([]);
    setActiveDraftIndex(0);
    setBatchSaveSummary(null);
    startExtract(nextFile);
  };

  const reset = () => {
    setFile(null);
    setPreview("");
    setResult(null);
    setMessage("");
    setSaveError("");
    setForm(EMPTY_FORM);
    setSaved(false);
    setIsDrag(false);
    setBatchDrafts([]);
    setBatchProgress("");
    setQueuedFiles([]);
    setActiveDraftIndex(0);
    setBatchSaveSummary(null);
  };

  const clearManualForm = () => {
    setManualForm(EMPTY_FORM);
    setManualError("");
  };

  const resetManual = () => {
    setManualSaved(false);
    setManualSavedData(null);
    setManualForm(EMPTY_FORM);
    setManualError("");
    setManualMessage("");
    setInputMethod("manual");
  };

  const changeInputMethod = (method) => {
    setInputMethod(method);
    setManualError("");
    setManualMessage("");
    setSaveError("");
    setBatchSaveSummary(null);
    setManualSaved(false);
    setManualSavedData(null);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDrag(false);
    const droppedFiles = Array.from(event.dataTransfer.files || []);
    if (queuedFiles.length) {
      addFilesToQueue(droppedFiles);
    } else {
      processFilesSequential(droppedFiles);
    }
  };

  const handleSave = async () => {
    if (!form.merchant.trim()) {
      setSaveError("Toko / Penerima wajib diisi.");
      return;
    }

    const amount = Number(form.amount);
    if (!amount || Number.isNaN(amount) || amount <= 0) {
      setSaveError("Nominal harus lebih dari 0.");
      return;
    }

    const activeResult = isBatchReview ? batchDrafts[activeDraftIndex]?.result : result;
    const source =
      activeResult?.document_type === "receipt"
        ? "receipt"
        : activeResult?.document_type === "screenshot"
        ? "screenshot"
        : docType === "receipt"
        ? "receipt"
        : "screenshot";

    setSaving(true);
    setSaveError("");
    try {
      await saveTransaction({
        merchant: form.merchant.trim(),
        amount,
        date: form.date || "",
        category: form.category || "lainnya",
        source,
        notes: form.notes || "",
      });
      if (isBatchReview) {
        setBatchDrafts((drafts) => drafts.map((draft, index) => (
          index === activeDraftIndex ? { ...draft, form, saved: true, saveError: "" } : draft
        )));
        setMessage("Transaksi ini berhasil disimpan.");
      } else {
        setSaved(true);
      }
    } catch (error) {
      const nextError = error.message || "Gagal menyimpan transaksi.";
      setSaveError(nextError);
      if (isBatchReview) {
        setBatchDrafts((drafts) => drafts.map((draft, index) => (
          index === activeDraftIndex ? { ...draft, saveError: nextError } : draft
        )));
      }
    } finally {
      setSaving(false);
    }
  };

  const handleSaveAllValid = async () => {
    if (!isBatchReview) return;
    setSaving(true);
    setSaveError("");
    setBatchSaveSummary(null);
    let savedCount = 0;
    let failedCount = 0;
    let totalAmount = 0;
    const savedItems = [];
    const nextDrafts = [...batchDrafts];

    for (let index = 0; index < nextDrafts.length; index += 1) {
      const draft = nextDrafts[index];
      const draftForm = draft.form || EMPTY_FORM;
      const amount = Number(draftForm.amount);
      if (!draftForm.merchant?.trim() || !amount || Number.isNaN(amount) || amount <= 0 || draft.saved) {
        continue;
      }
      const draftSource =
        draft.result?.document_type === "receipt"
          ? "receipt"
          : draft.result?.document_type === "screenshot"
          ? "screenshot"
          : docType === "receipt"
          ? "receipt"
          : "screenshot";
      try {
        await saveTransaction({
          merchant: draftForm.merchant.trim(),
          amount,
          date: draftForm.date || "",
          category: draftForm.category || "lainnya",
          source: draftSource,
          notes: draftForm.notes || "",
        });
        savedCount += 1;
        totalAmount += amount;
        savedItems.push({
          merchant: draftForm.merchant.trim(),
          amount,
          date: draftForm.date || "",
          category: draftForm.category || "lainnya",
        });
        nextDrafts[index] = { ...draft, saved: true, saveError: "" };
        setBatchDrafts([...nextDrafts]);
      } catch (error) {
        failedCount += 1;
        nextDrafts[index] = {
          ...draft,
          saveError: error.message || "Gagal menyimpan transaksi.",
        };
        setBatchDrafts([...nextDrafts]);
      }
    }

    setSaving(false);
    const skippedCount = batchDrafts.length - savedCount;
    if (savedCount > 0) {
      setBatchSaveSummary({
        savedCount,
        totalCount: batchDrafts.length,
        totalAmount,
        savedItems,
        skippedCount,
      });
      setMessage(failedCount > 0 ? `${failedCount} transaksi belum tersimpan. Cek kembali datanya.` : "");
    } else if (failedCount > 0) {
      setMessage(`${failedCount} transaksi belum tersimpan. Cek kembali datanya.`);
    } else {
      setMessage("Belum ada transaksi valid untuk disimpan. Lengkapi data terlebih dahulu.");
    }
  };

  const handleManualSave = async () => {
    if (!manualForm.merchant.trim()) {
      setManualError("Toko / Penerima wajib diisi.");
      return;
    }

    const amount = Number(manualForm.amount);
    if (!amount || Number.isNaN(amount) || amount <= 0) {
      setManualError("Nominal harus diisi dengan angka lebih dari 0.");
      return;
    }

    if (!manualForm.date) {
      setManualError("Tanggal wajib diisi.");
      return;
    }

    if (!manualForm.category) {
      setManualError("Kategori wajib dipilih.");
      return;
    }

    setSaving(true);
    setManualError("");
    setManualMessage("");
    try {
      await saveTransaction({
        merchant: manualForm.merchant.trim(),
        amount,
        date: manualForm.date,
        category: manualForm.category,
        source: "manual",
        notes: manualForm.notes || "",
      });
      setManualSavedData({
        merchant: manualForm.merchant.trim(),
        amount,
        date: manualForm.date,
        category: manualForm.category,
      });
      setManualForm(EMPTY_FORM);
      setManualSaved(true);
    } catch (error) {
      setManualError(error.message || "Transaksi belum berhasil disimpan. Coba lagi sebentar.");
    } finally {
      setSaving(false);
    }
  };

  const currentStatus = result ? statusLabel(result.status, form) : "";

  return (
    <>
      <Header
        title="Upload & Proses"
        subtitle="Tambahkan transaksi dengan scan struk, screenshot pembayaran, atau input manual."
      />

      <div className="page-body">
        {batchSaveSummary ? (
          <div className="card card-body" style={{ maxWidth: 640, margin: "0 auto", textAlign: "center" }}>
            <CheckCircle2 size={56} color="var(--color-success)" style={{ marginBottom: 14 }} />
            <div className="card-title" style={{ marginBottom: 6 }}>
              {batchSaveSummary.savedCount === batchSaveSummary.totalCount
                ? `${batchSaveSummary.savedCount} Transaksi Tersimpan`
                : `${batchSaveSummary.savedCount} dari ${batchSaveSummary.totalCount} Transaksi Tersimpan`}
            </div>
            {batchSaveSummary.skippedCount > 0 && (
              <div style={{ color: "var(--color-muted)", marginBottom: 12 }}>
                {batchSaveSummary.skippedCount} transaksi belum tersimpan. Cek kembali data yang masih kosong.
              </div>
            )}
            <div style={{ fontSize: 24, fontWeight: 850, color: "var(--color-primary)", marginBottom: 18 }}>
              Total tersimpan: {formatRp(batchSaveSummary.totalAmount)}
            </div>
            <div style={{ textAlign: "left", marginBottom: 22 }}>
              {batchSaveSummary.savedItems.map((item, index) => (
                <div
                  key={`${item.merchant}-${index}`}
                  className="flex items-center justify-between gap-3"
                  style={{ padding: "9px 0", borderBottom: "1px solid var(--color-border)" }}
                >
                  <span style={{ fontWeight: 700 }}>{index + 1}. {item.merchant}</span>
                  <span style={{ fontWeight: 800 }}>{formatRp(item.amount)}</span>
                </div>
              ))}
            </div>
            <div className="flex gap-3" style={{ justifyContent: "center" }}>
              <button className="btn btn-secondary" onClick={reset}>Upload Lagi</button>
              <button className="btn btn-primary" onClick={() => onNavigate("history")}>
                Lihat Riwayat
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        ) : saved ? (
          <div className="card card-body" style={{ maxWidth: 560, margin: "0 auto", textAlign: "center" }}>
            <CheckCircle2 size={56} color="var(--color-success)" style={{ marginBottom: 14 }} />
            <div className="card-title" style={{ marginBottom: 6 }}>Transaksi Tersimpan</div>
            <div style={{ color: "var(--color-muted)", marginBottom: 8 }}>{form.merchant}</div>
            <div style={{ fontSize: 30, fontWeight: 850, color: "var(--color-primary)", marginBottom: 22 }}>
              {formatRp(form.amount)}
            </div>
            <div className="flex gap-3" style={{ justifyContent: "center" }}>
              <button className="btn btn-secondary" onClick={reset}>Upload Lagi</button>
              <button className="btn btn-primary" onClick={() => onNavigate("history")}>
                Lihat Riwayat
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        ) : manualSaved && manualSavedData ? (
          <div className="card card-body" style={{ maxWidth: 560, margin: "0 auto", textAlign: "center" }}>
            <CheckCircle2 size={56} color="var(--color-success)" style={{ marginBottom: 14 }} />
            <div className="card-title" style={{ marginBottom: 6 }}>Transaksi Tersimpan</div>
            <div style={{ color: "var(--color-muted)", marginBottom: 8 }}>{manualSavedData.merchant}</div>
            <div style={{ fontSize: 30, fontWeight: 850, color: "var(--color-primary)", marginBottom: 22 }}>
              {formatRp(manualSavedData.amount)}
            </div>
            <div className="flex gap-3" style={{ justifyContent: "center" }}>
              <button className="btn btn-secondary" onClick={resetManual}>Input Lagi</button>
              <button className="btn btn-primary" onClick={() => onNavigate("history")}>
                Lihat Riwayat
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        ) : (
          <div className="upload-shell">
            <div className="input-method-tabs section-gap" role="tablist" aria-label="Metode input transaksi">
              {INPUT_METHODS.map(({ value, label, icon: Icon }) => (
                <button
                  key={value}
                  type="button"
                  role="tab"
                  aria-selected={inputMethod === value}
                  className={`method-tab${inputMethod === value ? " active" : ""}`}
                  onClick={() => changeInputMethod(value)}
                >
                  <Icon size={17} />
                  {label}
                </button>
              ))}
            </div>

            {inputMethod === "scan" ? (
              <>
                <div className="card card-body section-gap">
                  <div className="card-title" style={{ fontSize: 18 }}>Scan Transaksi</div>
                  <div style={{ color: "var(--color-muted)", fontSize: 14, marginBottom: 16 }}>
                    Unggah struk atau screenshot pembayaran, lalu cek hasilnya sebelum disimpan.
                  </div>
              <div className="card-title" style={{ fontSize: 16 }}>Jenis Transaksi</div>
              <div className="type-selector">
                {DOC_TYPES.map(({ value, label, icon: Icon }) => (
                  <button
                    key={value}
                    className={`type-option${docType === value ? " active" : ""}`}
                    onClick={() => setDocType(value)}
                  >
                    <Icon size={16} />
                    {label}
                  </button>
                ))}
              </div>
              <div style={{ marginTop: 12, color: "var(--color-muted)", fontSize: 14 }}>
                Biarkan Otomatis jika belum yakin jenis transaksinya.
              </div>
            </div>

            {!preview ? (
              <>
                <div
                  className={`upload-zone${isDrag ? " drag-over" : ""}`}
                  onClick={() => openFilePicker("single")}
                  onDragOver={(event) => {
                    event.preventDefault();
                    setIsDrag(true);
                  }}
                  onDragLeave={() => setIsDrag(false)}
                  onDrop={handleDrop}
                >
                  <div>
                    <div className="upload-zone-icon">
                      <UploadCloud size={28} />
                    </div>
                    <div className="upload-zone-title">Drag & drop struk atau screenshot di sini</div>
                    <div className="upload-zone-desc">atau klik untuk memilih file</div>
                    <div className="upload-zone-hint">Format PNG, JPG, atau JPEG hingga 10 MB</div>
                  </div>
                </div>
              <div className="flex gap-2" style={{ marginTop: 14, justifyContent: "center", flexWrap: "wrap" }}>
                <button className="btn btn-secondary btn-sm" type="button" onClick={() => openFilePicker("queue")}>
                  Tambah ke Batch
                </button>
                {queuedFiles.length > 0 && (
                  <>
                    <button
                      className="btn btn-primary btn-sm"
                      type="button"
                      onClick={() => processFilesSequential(queuedFiles)}
                      disabled={extracting}
                    >
                      {extracting ? "Sedang memproses..." : `Proses ${queuedFiles.length} Transaksi`}
                    </button>
                    <button className="btn btn-ghost btn-sm" type="button" onClick={clearQueue}>
                      Bersihkan Batch
                    </button>
                  </>
                )}
              </div>
                <div style={{ marginTop: 8, textAlign: "center", color: "var(--color-muted)", fontSize: 13.5 }}>
                  Maksimal 3 transaksi per proses. File diproses satu per satu.
                </div>
                {queuedFiles.length > 0 && (
                  <div className="alert alert-info mt-4">
                    <strong style={{ fontSize: 15, fontWeight: 700 }}>{queuedFiles.length} transaksi siap diproses.</strong>
                    <div style={{ marginTop: 6, fontSize: 14 }}>
                      {queuedFiles.map((queuedFile, index) => (
                        <div key={`${queuedFile.name}-${index}`}>{index + 1}. {queuedFile.name}</div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className={`upload-result-grid${extracting ? " processing" : ""}`}>
                <div className="card preview-card">
                  <div className="img-preview-wrap">
                    <img className="img-preview" src={preview} alt="Preview gambar transaksi" />
                  </div>

                  <div className="flex items-center justify-between gap-3" style={{ marginTop: 16, flexWrap: "wrap" }}>
                    {extracting ? (
                      <span className="detect-badge detect-badge-auto">
                        <ScanLine size={15} />
                        {docType === "auto" ? "Membaca jenis transaksi" : "Membaca transaksi"}
                      </span>
                    ) : result ? <DetectionBadge result={result} selectedType={docType} /> : null}
                    <div className="flex gap-2">
                      <button className="btn btn-ghost btn-sm" onClick={reset}>Ganti Gambar</button>
                      <button className="btn btn-secondary btn-sm" onClick={() => startExtract()} disabled={extracting}>
                        <RefreshCw size={15} />
                        {extracting ? "Sedang Diproses" : "Proses Ulang"}
                      </button>
                    </div>
                  </div>

                  {extracting && (
                    <div className="alert alert-info mt-4">
                      <div className="loading-bar">
                        <span className="spinner" />
                        <div>
                          <strong>Membaca transaksi...</strong>
                          <div>{batchProgress || "Biasanya membutuhkan beberapa detik."}</div>
                          {batchProgress && (
                            <div style={{ marginTop: 4, color: "var(--color-muted)", fontSize: 13 }}>
                              Mohon tunggu sampai proses saat ini selesai.
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  {batchDrafts.length > 1 && (
                    <div className="alert alert-info mt-4">
                      <strong style={{ fontSize: 15, fontWeight: 700 }}>
                        {batchDrafts.length} transaksi selesai dibaca. Pilih salah satu untuk cek detailnya.
                      </strong>
                      <div style={{ marginTop: 6, fontSize: 14 }}>
                        {batchDrafts.map(({ file: draftFile, form: draftForm, saved: draftSaved }, index) => (
                          <button
                            key={`${draftFile}-${index}`}
                            type="button"
                            className={`btn btn-ghost btn-sm${activeDraftIndex === index ? " active" : ""}`}
                            style={{
                              justifyContent: "flex-start",
                              width: "100%",
                              marginTop: 6,
                              fontWeight: activeDraftIndex === index ? 800 : 600,
                              background: activeDraftIndex === index ? "rgba(99, 102, 241, 0.10)" : "transparent",
                            }}
                            onClick={() => applyDraft(batchDrafts[index], index)}
                          >
                            {index + 1}. {draftForm.merchant || draftFile} {draftForm.amount ? `- ${formatRp(draftForm.amount)}` : ""}
                            {draftSaved ? " - Tersimpan" : ""}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {!extracting && result && (
                <div className="card card-body">
                  <div className="flex items-center justify-between gap-3" style={{ marginBottom: 16 }}>
                    <div>
                      <div className="card-title" style={{ marginBottom: 4 }}>Periksa & Sesuaikan</div>
                      <div style={{ color: "var(--color-muted)", fontSize: 14 }}>
                        Cek kembali hasil pembacaan sebelum disimpan.
                      </div>
                    </div>
                    <span className={`badge ${
                      currentStatus === "Terbaca"
                        ? "badge-green"
                        : currentStatus === "Belum terbaca"
                        ? "badge-red"
                        : "badge-yellow"
                    }`}>
                      {currentStatus}
                    </span>
                  </div>

                  {message && <div className="alert alert-warning mb-4">{message}</div>}

                  <div className="form-group">
                    <label className="form-label">Merchant</label>
                    <input
                      className="form-input"
                      value={form.merchant}
                      onChange={(event) => updateActiveDraftForm({ ...form, merchant: event.target.value })}
                      placeholder="Nama toko atau penerima"
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Nominal (Rp)</label>
                    <input
                      className="form-input"
                      type="number"
                      min="0"
                      value={form.amount}
                      onChange={(event) => updateActiveDraftForm({ ...form, amount: event.target.value })}
                      placeholder="0"
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Tanggal</label>
                    <input
                      className="form-input"
                      value={form.date}
                      onChange={(event) => updateActiveDraftForm({ ...form, date: event.target.value })}
                      placeholder="DD/MM/YYYY"
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Kategori</label>
                    <select
                      className="form-select"
                      value={form.category}
                      onChange={(event) => updateActiveDraftForm({ ...form, category: event.target.value })}
                    >
                      {categories.map((category) => (
                        <option key={category.value} value={category.value}>
                          {category.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Catatan Opsional</label>
                    <input
                      className="form-input"
                      value={form.notes}
                      onChange={(event) => updateActiveDraftForm({ ...form, notes: event.target.value })}
                      placeholder="Tambahkan catatan..."
                    />
                  </div>

                  {saveError && <div className="alert alert-error mt-4">{saveError}</div>}

                  <div className="divider" />

                  <div className="flex gap-2">
                    <button className="btn btn-primary" onClick={handleSave} disabled={saving || extracting}>
                      {saving ? (
                        <>
                          <span className="spinner" />
                          Menyimpan...
                        </>
                      ) : (
                        "Simpan Transaksi"
                      )}
                    </button>
                    {isBatchReview && (
                      <button className="btn btn-secondary" onClick={handleSaveAllValid} disabled={saving || extracting}>
                        Simpan Semua
                      </button>
                    )}
                    <button className="btn btn-ghost" onClick={reset}>Batal</button>
                  </div>
                </div>
                )}
              </div>
            )}
              </>
            ) : (
              <div className="manual-form-wrap">
                <div className="card card-body manual-form-card">
                  <div className="card-title" style={{ marginBottom: 4 }}>Input Manual Transaksi</div>
                  <div className="card-subtitle" style={{ marginTop: 0 }}>
                    Catat transaksi langsung tanpa membaca gambar.
                  </div>
                  <div className="alert alert-info mb-4">
                    Gunakan input manual jika transaksi sulit terbaca atau ingin mencatat transaksi tanpa gambar.
                  </div>

                  {manualError && <div className="alert alert-error mb-4">{manualError}</div>}

                  <div className="form-group">
                    <label className="form-label">Toko / Penerima</label>
                    <input
                      className="form-input"
                      value={manualForm.merchant}
                      onChange={(event) => setManualForm({ ...manualForm, merchant: event.target.value })}
                      placeholder="Contoh: Alfamart, Kopi Kenangan, GoBills PLN"
                    />
                  </div>

                  <div className="manual-form-grid">
                    <div className="form-group">
                      <label className="form-label">Nominal (Rp)</label>
                      <input
                        className="form-input"
                        type="number"
                        min="0"
                        value={manualForm.amount}
                        onChange={(event) => setManualForm({ ...manualForm, amount: event.target.value })}
                        placeholder="Contoh: 24000"
                      />
                    </div>

                    <div className="form-group">
                      <label className="form-label">Tanggal</label>
                      <input
                        className="form-input"
                        type="date"
                        value={manualForm.date}
                        onChange={(event) => setManualForm({ ...manualForm, date: event.target.value })}
                        placeholder="Contoh: 28/04/2026"
                      />
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Kategori</label>
                    <select
                      className="form-select"
                      value={manualForm.category}
                      onChange={(event) => setManualForm({ ...manualForm, category: event.target.value })}
                    >
                      {categories.map((category) => (
                        <option key={category.value} value={category.value}>
                          {category.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Catatan Opsional</label>
                    <input
                      className="form-input"
                      value={manualForm.notes}
                      onChange={(event) => setManualForm({ ...manualForm, notes: event.target.value })}
                      placeholder="Contoh: Makan siang / belanja bulanan"
                    />
                  </div>

                  <div className="divider" />

                  <div className="flex gap-2" style={{ flexWrap: "wrap" }}>
                    <button className="btn btn-primary" onClick={handleManualSave} disabled={saving}>
                      {saving ? (
                        <>
                          <span className="spinner" />
                          Menyimpan...
                        </>
                      ) : (
                        "Simpan Transaksi"
                      )}
                    </button>
                    <button className="btn btn-ghost" onClick={clearManualForm} disabled={saving}>
                      Bersihkan Form
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          multiple
          style={{ display: "none" }}
          onChange={(event) => {
            if (filePickMode === "queue") {
              addFilesToQueue(event.target.files);
            } else {
              processFilesSequential(event.target.files);
            }
            event.target.value = "";
          }}
        />
      </div>
    </>
  );
}
